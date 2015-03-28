import numpy as NP
import sklearn
import Features
import SymbolData


def train(model, training):
    model.fit(Features.features(training), SymbolData.classNumbers(training))
    return model

    
