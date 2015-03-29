# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""

# def normalize
def interp( x12, y12, t):
    x = x12[0] * (1- t) + x12[1] * t
    y = y12[0] * (1 -t) + y12[1] * t
    return (x, y)


import SymbolData
import matplotlib.pyplot as plt
import numpy as np

symbols = SymbolData.unpickleSymbols("train.dat")

xmin = min(symbols[0].xs())
ymin = min(symbols[0].ys())

scale = 49
plt.figure(1)
for i in range(len(symbols[0].strokes)):
    for j in range(len(symbols[0].strokes[i].xs)):
        symbols[0].strokes[i].xs[j] = (symbols[0].strokes[i].xs[j]-xmin)*scale
        symbols[0].strokes[i].ys[j] = (symbols[0].strokes[i].ys[j]-ymin)*scale
    plt.plot(symbols[0].strokes[i].xs,symbols[0].strokes[i].ys,'b')
plt.show()

xs = np.array([])
ys = np.array([])
for i in range(len(symbols[0].strokes)):
    for j in range(len(symbols[0].strokes[i].xs)-1):
        for t in np.linspace(0,1,30):
            x,y = interp(symbols[0].strokes[i].xs[j:j+2],symbols[0].strokes[i].ys[j:j+2],t)
            xs = np.append(xs,x)
            ys = np.append(ys,y)
#plt.show()

I = np.zeros((max(symbols[0].ys())+1,max(symbols[0].xs())+1))

plt.figure(2)
#for i in range(len(symbols[0].xs())):
#    I[max(symbols[0].ys())-int(symbols[0].ys()[i])][int(symbols[0].xs()[i])] = 1
for i in range(len(xs)):
    I[max(symbols[0].ys())-int(ys[i])][int(xs[i])] = 1
    
plt.imshow(I)
