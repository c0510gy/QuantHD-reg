import sys
import time
import math
import numpy as np
import joblib
from enum import Enum

from tqdm import tqdm_notebook

# dump basis and its param into a file, return the name of file
def saveEncoded(encoded, labels, id = "", data_type = "unknown"):
    filename = "encoded_%s_%s.pkl" % (id, data_type)
    sys.stderr.write("Dumping data into %s \n"%filename)
    joblib.dump((encoded, labels), open(filename, "wb"), compress=True)
    return filename

# Load basis from a file
def loadEncoded(filename):
    encoded, labels = joblib.load(filename)
    return encoded, labels

# Class: HD_encoder
# Use: take in a basis and a noise flag to create instance, call functions to with data to encode
class HD_encoder:
    def __init__(self, basis, noise=True):
        self.basis = basis
        self.D = basis.shape[0]
        self.noises = []
        if noise:
            self.noises = np.random.uniform(0, 2 * math.pi, self.D)
        else:
            self.noises = np.zeros(self.D)

    #encode one vector/sample into a HD vector
    def encodeDatum(self, datum):
        #encoded = np.empty(self.D)
        #for i in range(self.D):
        #    encoded[i] = np.cos(np.dot(datum,self.basis[i]) + self.noises[i]) * np.sin(np.dot(datum, self.basis[i]))
        encoded = np.matmul(self.basis, datum)
        encoded = np.cos(encoded)
        return encoded

    # encode data using the given basis
    # noise: default Gaussian noise
    def encodeData(self, data):
        start = time.time()
        sys.stderr.write("Encoding data of shape %s\n"%str(data.shape))
        assert data.shape[1] == self.basis.shape[1]
        noises = []
        encoded = []
        for i in tqdm_notebook(range(len(data)), desc='samples encoded'):
            encoded.append(self.encodeDatum(data[i]))
        end = time.time()
        sys.stderr.write("Time spent: %d sec\n" % int(end - start))
        return np.asarray(encoded)


