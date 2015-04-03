import sys
import pickle
import SymbolData
import Classification
import Features
from sklearn.metrics import accuracy_score

usage = "Usage: $ python test.py testingFilename stateFilename outdir"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) != 3): 
        print(("bad number of args:" , len(argv)))
    else:
        with open(argv[0], 'rb') as f:
            exprs, classes = pickle.load(f)
        with open(argv[1], 'rb') as f:
            model, pca =  pickle.load(f)


        #the following is a placeholder until I am sure we have propper analysis tools for evaluating our results if we preserve files.
        symbs = SymbolData.allSymbols(exprs)
        symbs = SymbolData.normalize(symbs, 99)
        f = Features.features(symbs)
        if (pca != None):
            f = pca.transform(f)
        pred = model.predict(f)
        
        print( "Accuracy on testing set : ", accuracy_score(SymbolData.classNumbers(symbs, classes), pred))

        #code to write out results goes here.
    

if __name__ == "__main__":
    sys.exit(main())
