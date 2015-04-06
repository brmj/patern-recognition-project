# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
#import numpy as NP
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw

#def getStatFeatures(symbol):
#    pts = NP.asarray(symbol[0].points()).T
#    f = NP.array([])
#    
#    if pts.shape[1] > 1:
#        cov = NP.cov(pts)
#        eig = NP.linalg.eig(cov)
#        ind = NP.argsort(eig[0])
#        eigVal = eig[0][ind]
#        eigVec = eig[1].T
#        eigVecSort = eigVec[ind]
#        f = NP.append(f,cov[0])
#        f = NP.append(f,cov[1][1])
#        f = NP.append(f,eigVal)
#        f = NP.append(f,eigVecSort)
#    else:
#        f = NP.zeros((9))
#    
#    return f

exprs , classes= SymbolData.unpickleSymbols("train.dat")
symbols = SymbolData.allSymbols(exprs)
scale = 99
symbols = SymbolData.normalize(symbols,scale)

#i=0
#for symbol in symbols:
#    print(i)
#    I = Features.features(symbol)
#    i+=1

#7989,12287,12288,23126,23127 test.dat
# 2467,3121,22071,22072,22731,46263 train.dat

# Without vertical repositioning
#6432,6433
i=0
for symbol in symbols[0:]:
    I = Features.symbolFeatures(symbol)
    print(i)
    i+=1

### Save FKI Testing features
#f = Features.features(symbols)
#Features.pickleFeatures(f,"FKIFeat_Test.dat")

### Save FKI Training features
#f = Features.features(symbols)
#Features.pickleFeatures(f,"FKIFeat_Train.dat")
#feat = Features.unpickleFeatures("FKIFeat_Train.dat")

### Save RWTH features
f = Features.features(symbols)
Features.pickleFeatures(f,"RWTHFeat_Train.dat")

### Save Statisticla features
#f = Features.features(symbols)
#Features.pickleFeatures(f,"StatFeat_Test.dat")
