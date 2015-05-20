import sys

import Segmentation
import SymbolData
import Parsing


usage = "Usage: $ python test_parsing.py outdir inkmldir [lgdir | vs | ls]"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) != 3): 
        print(("bad number of args:" , len(argv)))
        print(usage)
    else:
        print("reading inkml files.")
        outdir = argv[0]
        inkmldir = argv[1]
        if not( argv[2] == 'vs' or argv[2] == 'ls' or argv[2] == 'bad'):
            lgdir = argv[2]
            parts = Segmentation.readTruePartsDirectory(inkmldir, warn=False, lgdir = lgdir, calcInts = False)
        else:
            parts = Segmentation.readTruePartsDirectory(inkmldir, warn=False, calcInts = False)
            print("Parsing.")
            pfunc = None
            if argv[2] == 'vs':
                pfunc = Parsing.veryStupidParse
            elif argv[2] == 'ls':
                pfunc = Parsing.lessStupidParse
            elif argv[2] == 'bad':
                pfunc = Parsing.badParse
            for part in parts:
                part.relations = pfunc(part).lg_rel_lines()
        exprs = [part.toExpression() for part in parts] 
 
        #code to write out results goes here.
        print ("Writing LG files.")
        for expr in exprs:
            expr.writeLG(outdir )
            
if __name__ == "__main__":
    sys.exit(main())

