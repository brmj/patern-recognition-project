import sys
import pickle
import os
#from sklearn.externals import joblib
import SymbolData
import Classification
import Features
import Segmentation
import Parsing
from sklearn.metrics import accuracy_score

usage = "Usage: $ python test_all.py stateFilename outdir inkmldir"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) != 3): 
        print(("bad number of args:" , len(argv)))
        print(usage)
    else:

  


        
        #model, pca = joblib.load(argv[1]) 
        with open(argv[0], 'rb') as f:
            model, pca, keys, seg =  pickle.load(f)

        print ("Segmenting")
        #cleverpart = Segmentation.mkCleverPart(model, pca)
        mypart = Segmentation.mkMergeClassifiedPart(seg[0])
        #exprs = Segmentation.readAndSegmentDirectory(argv[2],  Segmentation.intersection_partition)
        #exprs = SymbolData.readAndSegmentDirectory(argv[2],  Segmentation.stupid_partition)
        #exprs = SymbolData.readAndSegmentDirectory(argv[2], cleverpart)
        #exprs = SymbolData.readInkmlDirectory(argv[2], argv[3])
        parts = Segmentation.readAndSegmentDirectory(argv[2], mypart, raw=True)
        
        
 
        #code to write out results goes here.
        print ("Classifying")
        parts = Classification.classifyPartitions(parts, keys, model, pca, showAcc = True)
        print ("Parsing")
        pfunc = Parsing.recursiveParse
        for part in parts:
            if (len(part.strokeGroups) != 0):
                part.relations = pfunc(part).lg_rel_lines()
            #print(part.relations)
        exprs = [part.toExpression() for part in parts] 
 
        #code to write out results goes here.
        print ("Writing LG files.")
        outdir = argv[1]
        for expr in exprs:
            expr.writeLG(outdir )
            
if __name__ == "__main__":
    sys.exit(main())
