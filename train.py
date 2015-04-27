import sys
import pickle
#from sklearn.externals import joblib
import SymbolData
import Classification
import Features
from sklearn.metrics import accuracy_score

usage = "Usage: $ python train.py (-nn|-rf|-et|modelFilename) outFilename (inFilename.dat | inkmldir lgdir)"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) < 3 or len (argv) > 4): 
        print(("bad number of args:" , len(argv)))
        print (usage)
    else:
        if (len ( argv ) == 3):
            exprs, keys = SymbolData.unpickleSymbols(argv[2])
        else:
            exprs = SymbolData.readInkmlDirectory(argv[2], argv[3])
            keys = SymbolData.defaultClasses
            
        if (argv[0] == "-nn" ):
            model = Classification.OneNN()
        elif (argv[0] == "-rf" ):
            model = Classification.makeRF()
        elif (argv[0] == "-et" ):
            model = Classification.makeET()
        else:
            with open(argv[0], 'rb') as f:
                model =  pickle.load(f)
                #this had better actually be a sklearn model or the equivelent.
                #things will break in ways that are hard for me to test for if it isn't.

        symbs = SymbolData.allSymbols(exprs)
        
        trained, pca = Classification.train(model, symbs, keys)

        print ("Done training.")
        if False:
            f = Features.features(symbs)
            if (pca != None):
                f = pca.transform(f)
            pred = model.predict(f)
            print( "Accuracy on training set : ", accuracy_score(SymbolData.classNumbers(symbs, keys), pred))


        seg = []
        #joblib.dump((trained, pca), argv[2])    
        with open(argv[1], 'wb') as f:
            pickle.dump((trained, pca, keys, seg), f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    sys.exit(main())
