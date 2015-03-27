import sys
import SymbolData


usage = "Usage: $ python split.py (file.inkml | directory) trainingFilename [testingFilename [trainingPerc]]"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.

    trainPerc = 2.0 / 3.0
    
    goodArgs = True
    if (len (argv) >= 2 and len (argv) <= 4):
        if (len (argv) == 2):
            trainPerc = 1.0
        if (len (argv) == 4):
            try:
                trainPerc = float(argv[3])
                if (trainPerc > 1 or trainPerc < 0):
                    goodArgs = False
            except ValueError:
                goodArgs = False
    else:
        print "bad number of args:" , len(argv)
        goodArgs = False

    if (goodArgs):
        symbols = SymbolData.readDirectory(argv[0])
        split = SymbolData.splitSymbols(symbols, trainPerc)
        SymbolData.pickleSymbols(split[0], argv[1])
        if (trainPerc != 1.0):
            SymbolData.pickleSymbols(split[1], argv[2])
        print "Split complete."
    else:
        print usage
    

if __name__ == "__main__":
    sys.exit(main())
