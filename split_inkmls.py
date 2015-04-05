import sys
import SymbolData


usage = "Usage: $ python split_inkmls.py (file.inkml | inkmlDirectory) lgdir trainingDir testingDir trainingLgDir testingLgDir [trainingPerc]]"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.

    trainPerc = 2.0 / 3.0
    
    goodArgs = True
    if (len (argv) >= 6 and len (argv) <= 7):
        if (len (argv) == 7):
            try:
                trainPerc = float(argv[6])
                if (trainPerc > 1 or trainPerc < 0):
                    goodArgs = False
            except ValueError:
                goodArgs = False
    else:
        print(("bad number of args:" , len(argv)))
        goodArgs = False

    if (goodArgs):

        inkmldir = argv[0]
        lgdir = argv[1]
        traindir = argv[2]
        testdir = argv[3]
        trainlg = argv[4]
        testlg = argv[5]
        
        SymbolData.splitFiles(inkmldir, lgdir, traindir, testdir, trainlg, testlg, trainPerc)

        print("Split complete.")
    else:
        print(usage)
    

if __name__ == "__main__":
    sys.exit(main())
