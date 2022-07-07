import sys
import time
import math
import numpy as np
import joblib
from enum import Enum

from tqdm import tqdm_notebook


# Generate one random vector of desired length and generation type
def generate_vector(vector_length, mu=0., sigma=1.):
    #if vector_type == "Gaussian":
    return np.random.normal(mu, sigma, vector_length)

# top/bottom left/right vectors, and kernel
def bak_extend(tl, tr, bl, br, h, w):
    old_d = tl.shape[0]
    new_d = (old_d - 1) // ((h * w)) + 1
    suffix = np.zeros(new_d * w * h - tl.shape[0])
    tl = np.concatenate((tl, suffix)).reshape((h, w, new_d))
    tr = np.concatenate((tr, suffix)).reshape((h, w, new_d))
    bl = np.concatenate((bl, suffix)).reshape((h, w, new_d))
    br = np.concatenate((br, suffix)).reshape((h, w, new_d))
    #print("tl shape" + str(tl.shape))

    extended = []
    for i in range(0, h + 1):
        for j in range(0, w + 1):
            tl_p = tl[:h - i, :w - j]
            tr_p = tr[:h - i, w - j:]
            bl_p = bl[h - i:, :w - j]
            br_p = br[h - i:, w - j:]
            top = None
            bottom = None
            cat = np.concatenate
            # The most disgusting code ever
            # Exist top
            if i != h:
                if j == 0:
                    top = tl_p
                elif j == w:
                    top = tr_p
                else:
                    top = cat((tl_p, tr_p), axis=1)
            # Exist bottom
            if i != 0:
                if j == 0:
                    bottom = bl_p
                elif j == w:
                    bottom = br_p
                else:
                    bottom = cat((bl_p, br_p), axis=1)
            if top is None:
                ext = bottom
            elif bottom is None:
                ext = top
            else:
                ext = cat((top, bottom), axis=0)
            ext = ext.reshape((-1))[:old_d]
            # print(i,j)
            # print(ext)
            extended.append(ext)
    extended = np.asarray(extended).reshape((h + 1, w + 1, -1))
    return extended

# h, w, d, k: height, weight, dimension, kernel size
def bak_layer(h, w, d, k, param):
    grid = []
    for i in range(math.ceil((h-1)/k)+1):
        line = []
        for j in range(math.ceil((w-1)/k)+1):
            line.append(generate_vector(d, param["vector"], param))
        grid.append(line)
    ext_grid = None
    for i in range(math.ceil((h-1)/k)):
        k_h = min(k, h-1-k*i)
        ext_line = None
        for j in range(math.ceil((w-1)/k)):
            k_w = min(k, w-1-k*j)
            extended = bak_extend(grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1], k_h, k_w)
            if ext_line is None:
                ext_line = extended
            else:
                ext_line = np.concatenate((ext_line[:, :-1], extended), axis = 1)
        if ext_grid is None:
            ext_grid = ext_line
        else:
            ext_grid = np.concatenate((ext_grid[:-1], ext_line), axis = 0)
    return ext_grid


# dump basis and its param into a file and return the name
def saveBasis(basis, param = None):
    if param is None:
        param = {"id": ""}
    filename = "base_%s.pkl" % param["id"]
    sys.stderr.write("Dumping basis into file: %s \n"%filename)
    joblib.dump((basis, param), open(filename, "wb"), compress=True)
    return filename

# Load basis from a file
def loadBasis(filename = "base_.pkl"):
    basis, param = joblib.load(filename)
    return basis, param

class HD_basis:

    def __init__(self, D, nFeatures):

        self.D = D
        self.nFeatures = nFeatures
        start = time.time()
        self.vanilla()
        end = time.time()
        sys.stderr.write('Encoding time: %s \n' % str(end - start))

    def vanilla(self):

        sys.stderr.write("Generating vanilla HD basis of shape... ")
        self.basis = []
        #for i in range(param["D"]):
        for _ in tqdm_notebook(range(self.D), desc='vectors'):
            self.basis.append(generate_vector(self.nFeatures))
        self.basis = np.asarray(self.basis)
        sys.stderr.write(str(self.basis.shape)+"\n")

    def getBasis(self):
        return self.basis
