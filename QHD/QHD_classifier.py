import sys
import random
import numpy as np
import copy
import sklearn
from scipy import stats
from numpy import dot, number
from numpy.linalg import norm



def sgn(i):
  if i > 0:
    return 1
  else:
    return -1

# n = e^-(|x|^2/(2std^2))
def gauss(x,y,std):
  n = np.linalg.norm(x - y)
  n = n ** 2
  n = n * -1
  n = n / (2 * (std**2))
  n = np.exp(n)
  return n

def poly(x,y,c,d):
  return (np.dot(x,y) + c) ** d

def binarize(X):
    return np.where(X >= 0, 1, -1)

# X should be the class matrix of shape nClasses * D
# 0 is mapped to most lowest value and 2^(bits)-1 highest
def quantize(X, bits):
    Nbins = 2**bits
    # ultimate cheess
    bins = [ (i/(Nbins)) for i in range(Nbins)]
   # notice the axis along which to normalize is always the last one
    nX = stats.norm.cdf(stats.zscore(X, axis = X.ndim-1))
    nX = np.digitize(nX, bins) - 1
    #print("Max and min bin value:", np.max(nX), np.min(nX))
    #print("Quantized from ", X)
    #print("To", nX)
    return nX

def cos_sim(vec1, vec2):

    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def hamming_map(vec1, vec2, std, mapping, pre_sampled):

    hamming_dist = 0.
    
    vec = np.abs(vec1 - vec2)
    if pre_sampled is not None:
        hamming_dist = np.sum(pre_sampled[vec.astype(int)])
    else:
        hamming_dist = sum([random.gauss(mu=mapping[int(v)], sigma=std) for v in vec])
    
    return -hamming_dist # invert distance so that larger is closer


class QHD_classifier:

    def __init__(self, D, nClasses, bits):

        self.D = D
        self.nClasses = nClasses
        self.bits = bits # if bits == -1, full precision
        self.classes = np.zeros((nClasses, D))
        self.quantized_classes = np.zeros((nClasses, D))

        self.pre_sampled = None

    def update(self, weight, guess, answer, lr):
        
        self.classes[guess]  -= lr * weight
        self.classes[answer] += lr * weight
    
    def predict_one_by_columns(self, query, number_of_cam_columns, std=None, mapping=None, use_pre_sampled=False, tn=3):

        qsample = quantize(query, self.bits) if self.bits > 0 else query
        dims_per_coulumn = self.D // number_of_cam_columns

        votes = np.zeros(self.nClasses, dtype=int)

        for c in range(number_of_cam_columns):
            left = dims_per_coulumn * c
            right = left + dims_per_coulumn
            sims = []
            for m in range(self.nClasses):
                val = 0.
                if mapping is None:
                    val = cos_sim(qsample[left:right], self.quantized_classes[m][left:right] if self.bits > 0 else self.classes[m][left:right])
                else:
                    val = hamming_map(qsample[left:right], self.quantized_classes[m][left:right] if self.bits > 0 else self.classes[m][left:right], std, mapping, self.pre_sampled if use_pre_sampled else None)

                sims.append(val)
            
            topn = np.argsort(sims)[-tn:]
            for i in topn: votes[i] += 1
        
        guess = np.argmax(votes)
        return guess

    def predict_one(self, query, std=None, mapping=None, use_pre_sampled=False, number_of_cam_columns='MAX', tn=3):

        #pre_sampled = None
        #if mapping is not None and use_pre_sampled:
        #    pre_sampled = np.array([random.gauss(mu=m, sigma=std) for m in mapping])

        if number_of_cam_columns != 'MAX':
            return self.predict_one_by_columns(query, number_of_cam_columns, std=std, mapping=mapping, use_pre_sampled=use_pre_sampled, tn=tn)

        qsample = quantize(query, self.bits) if self.bits > 0 else query
        guess, maxVal = -1, None
        for m in range(self.nClasses):
            val = 0.
            if mapping is None:
                val = cos_sim(qsample, self.quantized_classes[m] if self.bits > 0 else self.classes[m])
            else:
                if number_of_cam_columns == 'MAX':
                    val = hamming_map(qsample, self.quantized_classes[m] if self.bits > 0 else self.classes[m], std, mapping, self.pre_sampled if use_pre_sampled else None)
                else:
                    dims_per_coulumn = self.D // number_of_cam_columns
                    val = hamming_map(qsample, self.quantized_classes[m] if self.bits > 0 else self.classes[m], std, mapping, self.pre_sampled if use_pre_sampled else None)

            if maxVal is None or val > maxVal:
                guess, maxVal = m, val
        
        return guess

    def init_train(self, data, label, lr=1.):

        assert self.D == data.shape[1]

        r = list(range(data.shape[0]))
        random.shuffle(r)
        for i in r:
            sample = data[i]
            answer = label[i]

            self.classes[answer] += sample * lr
        
        return -1

    def random_bit_flip_by_gaussian(self, std):

        cnt_flipped, tot = 0, 0

        dist = 1 / (2**self.bits-1)

        for i in range(self.nClasses):
            for j in range(self.D):

                val = random.gauss(mu=self.quantized_classes[i, j] * dist, sigma=std)

                '''
                prv_class = max(0, self.quantized_classes[i, j] - 1)
                nxt_class = min(2**self.bits-1, self.quantized_classes[i, j] + 1)

                nxt_val = self.quantized_classes[i, j]

                if val < prv_class + 0.5:
                    nxt_val = prv_class
                elif val >= nxt_class - 0.5:
                    nxt_val = nxt_class
                # else => prv_class + 0.5 <= val < nxt_class - 0.5 => remains
                '''

                flipped_val = min(2**self.bits-1, max(0, (val + dist / 2) // dist)) + 0.0

                tot += 1
                if self.quantized_classes[i, j] != flipped_val:
                    cnt_flipped += 1

                self.quantized_classes[i, j] = flipped_val
                
        return cnt_flipped / tot

    def random_bit_flip_by_prob(self, prob_table):

        cnt_flipped, tot = 0, 0

        for i in range(self.nClasses):
            for j in range(self.D):

                prv_qval = max(0, self.quantized_classes[i, j] - 1)
                nxt_qval = min(2**self.bits-1, self.quantized_classes[i, j] + 1)

                r = random.random() * 100.

                flipped_val = self.quantized_classes[i, j]

                if r < prob_table[int(flipped_val)][1]:
                    flipped_val = prv_qval
                elif r < prob_table[int(flipped_val)][1] + prob_table[int(flipped_val)][0]:
                    flipped_val = nxt_qval
                
                tot += 1
                if self.quantized_classes[i, j] != flipped_val:
                    cnt_flipped += 1

                self.quantized_classes[i, j] = flipped_val
                
        return cnt_flipped / tot

    def model_projection(self):

        if self.bits == -1:

            return -1

        for i in range(self.nClasses):
            
            self.quantized_classes[i] = quantize(self.classes[i], self.bits)
        
        return -1
    
    def gen_pre_sampled(self, std, mapping):

        self.pre_sampled = np.array([random.gauss(mu=m, sigma=std) for m in mapping])

    def itr_train(self, data, label, lr=1., std=None, mapping=None, use_pre_sampled=False):

        assert self.D == data.shape[1]

        correct, total = 0, 0

        r = list(range(data.shape[0]))
        random.shuffle(r)
        for i in r:
            sample = data[i]
            answer = label[i]

            guess = self.predict_one(sample, std=std, mapping=mapping, use_pre_sampled=use_pre_sampled)
            if guess == answer:
                correct += 1
            total += 1
            
            self.update(sample, guess, answer, lr)
        
        return correct / total

    def predict(self, data, std=None, mapping=None, use_pre_sampled=False):

        assert self.D == data.shape[1]

        prediction = []

        for i in range(data.shape[0]):
            guess = self.predict_one(data[i], std=std, mapping=mapping, use_pre_sampled=use_pre_sampled)
            prediction.append(guess)
        
        return prediction

    def test(self, data, label, std=None, mapping=None, use_pre_sampled=False, number_of_cam_columns='MAX', tn=3):

        assert self.D == data.shape[1]

        correct, total = 0, 0
        for i in range(data.shape[0]):
            answer = label[i]
            guess = self.predict_one(data[i], std=std, mapping=mapping, use_pre_sampled=use_pre_sampled, number_of_cam_columns=number_of_cam_columns, tn=tn)
            if guess == answer:
                correct += 1
            total += 1
        
        return correct / total
