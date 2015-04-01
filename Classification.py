import numpy as NP
import sklearn
import Features
import SymbolData
import matplotlib.pyplot as plt

def train(model, training):
    model.fit(Features.features(training), SymbolData.classNumbers(training))
    return model

symbols = SymbolData.unpickleSymbols("train.dat")
symbols = SymbolData.normalize(symbols,99)


Features.showImg(symbols[3])
x = Features.normalize(symbols[0],99)

f = Features.features(symbols[0:2])
