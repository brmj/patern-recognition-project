import sys
import pickle
import os
#from sklearn.externals import joblib
import SymbolData
import Classification
import Features
import Segmentation
from sklearn.metrics import accuracy_score

usage = "Usage: $ python test.py stateFilename outdir inkmldir"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) != 3): 
        print(("bad number of args:" , len(argv)))
        print(usage)
    else:

  


        
        #model, pca = joblib.load(argv[1]) 
        with open(argv[0], 'rb') as f:
            model, pca, keys =  pickle.load(f)

        print ("Segmenting")
        cleverpart = Segmentation.mkCleverPart(model, pca)
        #exprs = SymbolData.readAndSegmentDirectory(argv[2],  Segmentation.intersection_partition)
        #exprs = SymbolData.readAndSegmentDirectory(argv[2],  Segmentation.stupid_partition)
        exprs = SymbolData.readAndSegmentDirectory(argv[2], cleverpart)
        #exprs = SymbolData.readInkmlDirectory(argv[2], argv[3])

        #the following is a placeholder until I am sure we have propper analysis tools for evaluating our results if we preserve files.
#        symbs = SymbolData.allSymbols(exprs)
#        print("Normalizing")
#        symbs = SymbolData.normalize(symbs, 99)
#        print("Calculating features.")
#        f = Features.features(symbs)
#        if (pca != None):
#            print ("doing PCA")
#            f = pca.transform(f)
#        print ("Classifying.")
#        pred = model.predict(f)
        
#        print( "Accuracy on testing set : ", accuracy_score(SymbolData.classNumbers(symbs, classes), pred))

        #code to write out results goes here.
        print ("Classifying")
        truths, preds = Classification.classifyExpressions(exprs, keys, model, pca, showAcc = True)
        print ("Writing LG files.")
        i = 0
        for expr in exprs:
            #if (preds[i] != -1): 
            f = (lambda p: keys[p])
            #    expr.classes = map (f, preds[i])

            expr.writeLG(argv[1],clss =  map (f, preds[i]) )
            i = i + 1
            
if __name__ == "__main__":
    sys.exit(main())
