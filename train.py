import sys
import pickle
import SymbolData
import Classification
import Features
from sklearn.metrics import accuracy_score

usage = "Usage: $ python train.py trainingFilename (-nn|-rf|modelFilename) outFilename"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) != 3): 
        print(("bad number of args:" , len(argv)))
    else:
        exprs, classes = SymbolData.unpickleSymbols(argv[0])

        if (argv[1] == "-nn" ):
            model = Classification.OneNN()
        elif (argv[1] == "-rf" ):
            model = Classification.makeRF()
        else:
            with open(argv[1], 'rb') as f:
                model =  pickle.load(f)
                #this had better actually be a sklearn model or the equivelent.
                #things will break in ways that are hard for me to test for if it isn't.

        symbs = SymbolData.allSymbols(exprs)
        trained, pca = Classification.train(model, symbs, classes)

        print ("Done training.")
        if True:
            f = Features.features(symbs)
            if (pca != None):
                f = pca.transform(f)
            pred = model.predict(f)
            print( "Accuracy on training set : ", accuracy_score(SymbolData.classNumbers(symbs, classes), pred))

        with open(argv[2], 'wb') as f:
            pickle.dump((trained, pca), f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    sys.exit(main())
