# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
#import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw

exprs , classes= SymbolData.unpickleSymbols("train.dat")
symbols = SymbolData.allSymbols(exprs)
scale = 99
symbols = SymbolData.normalize(symbols,scale)

#i=0
#for symbol in symbols:
#    print(i)
#    I = Features.getImg(symbol)
#    i+=1

#I = Image.new("L",(round(max(symbol.xs()))+1,round(max(symbol.ys()))+1))
#for stroke in symbol.strokes:
#    p = stroke.asPoints()
#    draw = ImageDraw.Draw(I)
#    draw.line(p,fill=255)  
#img = NP.asarray(list(I.getdata()))
#img = NP.reshape(img,(I.size[1],I.size[0]))

### Save FKI Testing features
#f = Features.features(symbols)
#Features.pickleFeatures(f,"FKIFeat_Test.dat")

### Save FKI Training features
f = Features.features(symbols)
Features.pickleFeatures(f,"FKIFeat_Train.dat")
#feat = Features.unpickleFeatures("FKIFeat_Train.dat")
