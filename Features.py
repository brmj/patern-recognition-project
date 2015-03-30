import numpy as NP
import SymbolData


# This is a skeleton of a file that will contain functions for various features.

def features(symbols):
    return map ( (lambda symbol: symbolFeatures(symbol)), symbols)

def symbolFeatures(symbol):
    f = []

    #Call feature functions here like so:
    f = f + xmean(symbol)
    f = f + ymean(symbol)
    f = f + xvar(symbol)
    f = f + yvar(symbol)    
    return f

# Some really simple global properties to start us off.
    
def xmean(symbol):
    return [NP.mean(symbol.xs())]

def ymean(symbol):
    return [NP.mean(symbol.ys())]

def xvar(symbol):
    return [NP.var(symbol.xs())]

def yvar(symbol):
    return [NP.var(symbol.ys())]

## FKI Features 
def getFKIfeatures(I):
    [W,H] = I.shape
    c4ant = H+1
    c5ant = 0
    f = NP.zeros((W,9)) # 
    for x in range(W):
        c1=0; c2=0; c3=0; c4=H+1; c5=0; c6=H+1; c7=0; c8=0; c9=0;
        for y in range(H):
            if(I[x][y]==1):
                c1+=1       # number of white pixels in the column
                c2+=y       # center of gravity of the column
                c3+=y**2    # second order moment of the column
                if(y<c4):
                    c4=y    # position of the upper contour in the column
                if(y>c5):
                    c5=y    # position of the lower contour in the column
            if(y>1 & I[x][y-1]!= I[x][y-2]):
                c8+=1       # Number of black-white transitions in the column
        
        c2 /= H
        c3 /= H**2
        
        for y in range(c4+1,c5):
            if(I[x][y]==1):
                c9+=1       # Number of white pixels between the upper and lower contours
        
        if(x+1<W):
            for y in range(H):
                if(I[x+1][y-1]==1):
                    if(y<c6):
                        c6=y     
                    if(y>c7):
                        c7=y
                    
        c6 = (c6-c4ant)/2 # Orientation of the upper contour in the column
        c7 = (c7-c5ant)/2 # Orientation of the lower contour in the column
        c4ant = c4
        c5ant = c5
        f[x] = NP.array([c1,c2,c3,c4,c5,c6,c7,c8,c9])
    return f
    