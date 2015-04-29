import numpy as NP
#import SymbolData
#from skimage.morphology import disk, binary_closing
#from skimage.filter import rank
#from skimage.transform import rescale
from sklearn import preprocessing
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw
import pickle

# This is a skeleton of a file that will contain functions for various features.

def features(strokeGroupPairs):
    return list(map ( (lambda sgpair: sgPairFeatures(sgpair)), strokeGroupPairs) )

# Get the features from a symbol
def sgPairFeatures(sgPair):
    sg1, sg2 = sgPair
    f = NP.array([])
    
    #Call feature functions here like so:
    f = NP.append(f, xmeanDist(sg1, sg2))
    f = NP.append(f, ymeanDist(sg1, sg2))
    f = NP.append(f, numstrokes(sg1, sg2))
    # f = NP.append(f, totlen(symbol))
    f = NP.append(f, width_ratio(sg1, sg2))
    f = NP.append(f, height_ratio(sg1, sg2))
    f = NP.append(f, horizontal_bounding_distance(sg1, sg2))    
    #f = NP.append(f, getStatFeatures(symb1))
    #f = NP.append(f, getStatFeatures(symb2))
    
    #the minimum, basic scaling needed for many classifiers to work corectly.
    #f_scaled = preprocessing.scale(f)
    # would have put PCA here, but that doesn't play nice with the way it needs to be used.
    #return f_scaled
    return f
    
# Some really simple global properties to start us off.
    
def xmeanDist(sg1, sg2):
    #print (NP.mean(sg1.xs()))
    return [NP.mean(sg1.xs()) - NP.mean(sg2.xs())]

def ymeanDist(sg1, sg2):
    return [NP.mean(sg1.ys()) - NP.mean(sg2.ys())]

def numstrokes(sg1, sg2):
    return [sg1.strokenum() + sg2.strokenum()]

def totlen(sg1, sg2):
    for sg in [sg1, sg2]:
        assert (not(sg.totlen() is None and len(sg.points()) >1))
    return [sg1.totlen() + sg2.totlen()]

def width_ratio(sg1, sg2):
    width1 = sg1.xmax() - sg1.xmin()
    width2 = sg2.xmax() - sg2.xmin()
    if (width2 < 0.01):
        width2 = 0.01 
    return [NP.divide(width1, width2)]

def height_ratio(sg1, sg2):
    height1 = sg1.ymax() - sg1.ymin()
    height2 = sg2.ymax() - sg2.ymin()
    if (height2 < 0.01):
        height2 = 0.01
    return [NP.divide(height1, height2)]

def horizontal_bounding_distance(sg1, sg2):
    #width1 = sg1.xmax - sg1.xmin
    #width2 = sg2.xmax - sg2.xmin
    #r12_x = NP.absolute(sg1.xmin + sg1.xmax - sg2.xmin - sg2.xmax)
    #return [NP.divide(r12_x, (width1 + width2))]
    #I'm not sure what you thought this was doing, but it isn't doing what we discussed.
    #print (sg1.xmin() , ' : ', sg1.xmax() , ', ', sg2.xmin() , ' : ', sg2.xmax())

    if (sg1.xmax() > sg2.xmax()):
        sgr = sg1
        sgl = sg2
    else:
        sgr = sg2
        sgl = sg1

        
    if(sgr.xmin() > sgl.xmax()):
        num = sgr.xmin() - sgl.xmax()
        denom = sgr.xmax() - min(sg1.xmin(), sg2.xmin())
        if (denom < 0.01):
            denom = 0.01
        return [NP.divide(num, denom)]
    else:
        #print ("Overlaps.")
        return [0]

    

                     
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


