# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
import matplotlib.pyplot as plt

      
symbols = SymbolData.unpickleSymbols("train.dat")
scale = 99
symbols = Features.normalize(symbols,scale)
symbols[0].plot()
plt.grid('on')

I = Features.getImg(symbols[0])
    
plt.imshow(I)
plt.gray()
plt.show()

f = Features.getFKIfeatures(I)
fki = Features.getMeanStd(f)
