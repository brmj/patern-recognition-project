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

