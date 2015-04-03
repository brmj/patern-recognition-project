import numpy as NP
import sklearn
import sklearn.ensemble
import Features
import SymbolData
import matplotlib.pyplot as plt

#1-NN classifier. Pretends to be a sklearn model, so we can use the same code.
class OneNN:
    def __init__(self):
        self.classes = NP.array([])
        self.features = NP.array([])

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        return self

    def predict(self, samples):
        return NP.array(list( map ((lambda s: self.nearest(s)), samples))) #have no idea why it breaks if I go straight to array.

    def nearest(self, sample):
        self.sqrs = NP.power(self.features - sample, 2)
        self.sums = NP.sum(self.sqrs, axis=1)
        self.dists = NP.sqrt(self.sums)
        self.idx = NP.argmin(self.dists)
        return self.classes[self.idx]

def makeRF():
    #play with the options once we have a reasonable set of features to experiment with.
    return sklearn.ensemble.RandomForestClassifier()
        
def train(model, training, keys= None):
    if model == "1nn":
        model = OneNN()
    elif model == "rf":
        model = makeRF()
    model.fit(Features.features(training), SymbolData.classNumbers(training, keys))
    return model

"""


symbols = SymbolData.unpickleSymbols("train.dat")
symbols = SymbolData.normalize(symbols,99)

Features.showImg(symbols[3])

f = Features.features(symbols[0:2])
 """
