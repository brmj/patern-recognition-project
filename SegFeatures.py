import numpy as NP
#import SymbolData
from skimage.morphology import disk, binary_closing
from skimage.filter import rank
#from skimage.transform import rescale
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle

# This is a skeleton of a file that will contain functions for various features.

def features(strokeGroupPairs):
    return list(map ( (lambda sgpair: sgPairFeatures(sgpair)), strokeGroupPairs))

# Get the features from a symbol
def sgPairFeatures(sgPair):
    sg1, sg2 = sgpair
    symb1 =sg1.toSymbol()
    symb2 = sg2.toSymbol()
    f = NP.array([])
    
    #Call feature functions here like so:
    f = NP.append(f,xmean(symbol))
    f = NP.append(f,ymean(symbol))
    f = NP.append(f,numstrokes(symbol))    
   
    f = NP.append(f,totlen(symbol))

    f = NP.append(f,getStatFeatures(symbol))
    
    #the minimum, basic scaling needed for many classifiers to work corectly.
    f_scaled = preprocessing.scale(f)
    # would have put PCA here, but that doesn't play nice with the way it needs to be used.
    return f_scaled

# Some really simple global properties to start us off.
    
def xmeanDist(sgPair):
    return [NP.mean(symbol.xs())]

def ymean(symbol):
    return [NP.mean(symbol.ys())]


    
def numstrokes(symbol):
    return[symbol.strokenum]

def totlen(symbol):
    assert(not(symbol.totlen() is None and len(symbol.points()) >1))
    return [symbol.totlen()]


def getStatFeatures(symbol):
    pts = NP.asarray(symbol.points()).T
    f = NP.array([])
    
    if pts.shape[1] > 1:
        cov = NP.cov(pts)
        eig = NP.linalg.eig(cov)
        ind = NP.argsort(eig[0])
        eigVal = eig[0][ind]
        eigVec = eig[1].T
        eigVecSort = eigVec[ind]
        f = NP.append(f,cov[0])
        f = NP.append(f,cov[1][1])
        f = NP.append(f,eigVal)
        f = NP.append(f,eigVecSort)
    else:
        f = NP.zeros((9))
    
    return f


def pickleFeatures(feat, filename):
    with open(filename, 'wb') as f:
        pickle.dump(feat, f, pickle.HIGHEST_PROTOCOL)
        #note that this may cause problems if you try to unpickle with an older version.


def unpickleFeatures(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


