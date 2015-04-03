import sys
import SymbolData


usage = "Usage: $ python split.py (file.inkml | directory) lgdir trainingFilename [testingFilename [trainingPerc]]"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.

    trainPerc = 2.0 / 3.0
    
    goodArgs = True
    if (len (argv) >= 3 and len (argv) <= 5):
        if (len (argv) == 3):
            trainPerc = 1.0
        if (len (argv) == 5):
            try:
                trainPerc = float(argv[4])
                if (trainPerc > 1 or trainPerc < 0):
                    goodArgs = False
            except ValueError:
                goodArgs = False
    else:
        print(("bad number of args:" , len(argv)))
        goodArgs = False

    if (goodArgs):
        exprs = SymbolData.readInkmlDirectory(argv[0], argv[1])
        classes = SymbolData.exprClasses(exprs)
        split = SymbolData.splitExpressions(exprs, trainPerc)
        SymbolData.pickleSymbols((split[0], classes), argv[2])
        if (trainPerc != 1.0):
            SymbolData.pickleSymbols((split[1], classes), argv[3])
        print("Split complete.")
    else:
        print(usage)
    

if __name__ == "__main__":
    sys.exit(main())
