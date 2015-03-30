# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
import matplotlib.pyplot as plt
import numpy as np

#import skimage
from skimage.morphology import disk
from skimage.filter import rank
#from skimage import data, filter

def interp( x12, y12, t):
    x = x12[0] * (1- t) + x12[1] * t
    y = y12[0] * (1 -t) + y12[1] * t
    return (x, y)

def normalize(symbols,scale):
    i=0
    for symbol in symbols:
        xmin = min(symbol.xs())
        ymin = min(symbol.ys())
        for i in range(len(symbol.strokes)):
            for j in range(len(symbol.strokes[i].xs)):
                symbol.strokes[i].xs[j] = (symbol.strokes[i].xs[j]-xmin)*scale/2
                symbol.strokes[i].ys[j] = (symbol.strokes[i].ys[j]-ymin)*scale/2
        symbols[i] = symbol
        i+=1
    return(symbols)

def getImg(symbol):
       xs = np.array([])
       ys = np.array([])

       for i in range(len(symbol.strokes)):
           for j in range(len(symbol.strokes[i].xs)-1):
               for t in np.linspace(0,1,30):
                   x,y = interp(symbol.strokes[i].xs[j:j+2],symbol.strokes[i].ys[j:j+2],t)
                   xs = np.append(xs,x)
                   ys = np.append(ys,y)
       I = np.zeros((max(symbol.ys())+1,max(symbol.xs())+1))

       for i in range(len(xs)):
           I[max(symbol.ys())-int(ys[i])][int(xs[i])] = 1

       I = rank.mean(I, selem=disk(1))
       
       for i in range(I.shape[0]):
           for j in range(I.shape[1]):
               if(I[i][j]>0.5):
                   I[i][j]=1
       return I
      
      
symbols = SymbolData.unpickleSymbols("train.dat")
scale = 99
symbols = normalize(symbols,scale)
symbols[0].plot()
plt.grid('on')

I = getImg(symbols[6])
    
plt.imshow(I)
plt.gray()
plt.show()

f = Features.getFKIfeatures(I)
