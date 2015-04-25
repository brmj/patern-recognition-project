import numpy as NP
import sklearn
import sklearn.ensemble
import sklearn.decomposition
import Features
import SymbolData
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from functools import reduce


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
    return sklearn.ensemble.RandomForestClassifier(n_estimators=20, n_jobs = -1, verbose=1)

def makeET():
    #play with the options once we have a reasonable set of features to experiment with.
    return sklearn.ensemble.ExtraTreesClassifier()
        
def train(model, training, keys, pca_num=None):
    if model == "1nn":
        model = OneNN()
    elif model == "rf":
        model = makeRF()
    training = SymbolData.normalize(training, 99)
    f = Features.features(training)
    pca = None
    if (pca_num != None):
        pca = sklearn.decomposition.PCA(n_components=pca_num)
        pca.fit(f)
        f = pca.transform(f)
    model.fit(Features.features(training), SymbolData.classNumbers(training, keys))
    return (model, pca)


def classifyExpressions(expressions, keys, model, pca, renormalize=True, showAcc = False):
    #this sort of does double duty, since it is both classifying the symbols
    # with side effects and returning stuff to evaluate the results.
    # Bad style. Sorry.

    cors = list([])
    preds = list([])
    tot = len(expressions)
    i = 0
    
#    s = showAcc
    
    for expr in expressions:
        correct, predicted =  classifyExpression(expr, keys, model, pca, renormalize)
        #assert (len(correct) == len(predicted))
#        if s:
#            cors = cors + [correct]
#        if correct == None:
#            s = False
#            cors = [[], []]
        
        preds = preds + [predicted]
        print(i, "/",tot)
        i+=1
        #print (correct, " -> ", predicted)

        
#    if s :
#        print (cors)
#        print( "Accuracy on testing set : ", accuracy_score(NP.concatenate(cors), NP.concatenate(preds)))
    return (cors, preds)
    
def classifyExpression(expression, keys, model, pca, renormalize=True):
    symbs = expression.symbols
    if renormalize:
        symbs = SymbolData.normalize(symbs, 99)
    f = Features.features(symbs)
    if (len (symbs) == 0):
        print(expression.name, " has no valid symbols!")
        return ([], [])
    if (pca != None):
        f = pca.transform(f)
    pred = model.predict(f)
    assert (max(pred) < len(keys))
    f = (lambda p: keys[p])
    expression.classes = map (f, pred)
    return (NP.array(SymbolData.classNumbers(symbs, keys)), pred)

"""


symbols = SymbolData.unpickleSymbols("train.dat")
symbols = SymbolData.normalize(symbols,99)

Features.showImg(symbols[3])

f = Features.features(symbols[0:2])
 """
