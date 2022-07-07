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

def hamming_map(vec1, vec2, std, mapping):

    hamming_dist = 0.
    
    vec = np.abs(vec1 - vec2)
    
    hamming_dist = sum([random.gauss(mu=mapping[abs(int(v))], sigma=std) for v in vec])
    
    return hamming_dist


class QHD_reg:

    def __init__(self, D, bits):

        self.D = D
        self.bits = bits # if bits == -1, full precision
        self.model = np.zeros((D))
        self.quantized_model = np.zeros((D))

        self.pre_sampled = None

    def update(self, query, pred_y, target_y, lr):
        
        self.model += lr * (target_y - pred_y) * query

    def predict(self, query, std=None, mapping=None):

        qsample = quantize(query, self.bits) if self.bits > 0 else query

        if self.bits == -1:
            return np.dot(qsample, self.model)

        if mapping is None:
            pred_y = np.dot(qsample, self.quantized_model)
        else:
            pred_y = hamming_map(qsample, self.quantized_model, std, mapping)
        
        return pred_y

    def random_bit_flip_by_prob(self, prob_table):

        cnt_flipped, tot = 0, 0

        for j in range(self.D):

            prv_qval = max(0, self.quantized_model[j] - 1)
            nxt_qval = min(2**self.bits-1, self.quantized_model[j] + 1)

            r = random.random() * 100.

            flipped_val = self.quantized_model[j]

            if r < prob_table[int(flipped_val)][1]:
                flipped_val = prv_qval
            elif r < prob_table[int(flipped_val)][1] + prob_table[int(flipped_val)][0]:
                flipped_val = nxt_qval
            
            tot += 1
            if self.quantized_model[j] != flipped_val:
                cnt_flipped += 1

            self.quantized_model[j] = flipped_val
                
        return cnt_flipped / tot

    def model_projection(self):

        if self.bits == -1:

            return -1
        
        self.quantized_model = quantize(self.model, self.bits)

        return -1
    
    def itr_train(self, data, targets, lr=1., std=None, mapping=None):

        assert self.D == data.shape[-1]

        mse_err = 0
        tot = 0

        r = list(range(data.shape[0]))
        random.shuffle(r)
        for i in r:
            sample = data[i]
            target_y = targets[i]

            pred_y = self.predict(sample, std=std, mapping=mapping)
            #print(sample, target_y)
            #print(pred_y)
            mse_err += (target_y - pred_y)**2
            tot += 1
            
            self.update(sample, pred_y, target_y, lr)
        
        return mse_err / tot

    def test(self, data, targets, std=None, mapping=None):

        assert self.D == data.shape[-1]

        mse_err = 0
        tot = 0

        for i in range(data.shape[0]):
            target_y = targets[i]
            pred_y = self.predict(data[i], std=std, mapping=mapping)
            
            mse_err += (target_y - pred_y)**2
            tot += 1
        
        return mse_err / tot
