import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
from pylab import *
import os
import shutil
import re
import random
import numpy.random
import scipy.stats
import pickle
import functools
import itertools
from functools import reduce


""" Contains representations for the relevant data,
    As well as functions for reading and processing it. """

class Stroke:
    """Represents a stroke as an n by 2 matrix, with the rows of
      the matrix equivelent to points from first to last. """
    def __init__(self, points, flip = False, ident = None):
        self.ident = ident
        self.xs = []
        self.ys = []
        for point in points:
            self.addPoint(point, flip)

    def plot(self, show = True, clear = True):
        if clear:
           PLT.clf()
        PLT.plot(self.xs, self.ys, 'ko-' )
           
        if show:
            PLT.show()

    def addPoint(self, point, flip = False):
        self.xs.append(point[0])
        if flip:
            self.ys.append(-1 * point[1])
        else:
            self.ys.append(point[1])

    def asPoints(self):
        return (list(zip(self.xs, self.ys)))

    def scale(self, xmin, xmax, ymin, ymax, xscale, yscale):
        if (xmax != xmin):
            self.xs = list(map( (lambda x: xscale * ((x - xmin) * 1.0 / (xmax - xmin))), self.xs))
        else:
            self.xs = list(map( (lambda x: 0), self.xs))
        if (ymax != ymin):
            self.ys = list(map( (lambda y: yscale * ((y - ymin) * 1.0 / (ymax - ymin))), self.ys))
        else:
            self.ys = list(map( (lambda y: 0), self.ys))
        self.xs = list(map( (lambda x: (x * 2) - xscale), self.xs))
        self.ys = list(map( (lambda y: (y * 2) - yscale), self.ys))

    def xmin(self):
        return min(self.xs)

    def xmax(self):
        return max(self.xs)

    def ymin(self):
        return min(self.ys)

    def ymax(self):
        return max(self.ys)
        
    def __str__(self):
        return 'Stroke:\n' + str(self.asPoints())

    
    
class Symbol:
    """Represents a symbol as a list of strokes. """
    def __init__(self, strokes, correctClass = None, norm = True, ident = None):
        self.strokes = strokes
        self.ident = ident
        if norm:
            self.normalize()
        self.correctClass = correctClass

    def plot(self, show = True, clear = True):
        if clear:
            PLT.clf()
        for stroke in self.strokes:
            stroke.plot(show = False, clear = False)
        if show:
            PLT.show()

    def xmin(self):
        return min(list(map( (lambda stroke: stroke.xmin()), self.strokes)))

    def xmax(self):
        return max(list(map( (lambda stroke: stroke.xmax()), self.strokes)))

    def ymin(self):
        return min(list(map( (lambda stroke: stroke.ymin()), self.strokes)))

    def ymax(self):
        return max(list(map( (lambda stroke: stroke.ymax()), self.strokes)))

    def points(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.asPoints()), self.strokes))), [])

    def xs(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.xs), self.strokes))), [])

    def ys(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.ys), self.strokes))), [])
    
    def normalize(self):

        self.xscale = 1.0
        self.yscale = 1.0
        self.xdif = self.xmax() - self.xmin()
        self.ydif = self.ymax() - self.ymin()
        #look out for a divide by zero here.
        #Would fix it, but still not quite sure what the propper way to handel it is.
        if (self.xdif > self.ydif):
            self.yscale = (self.ydif * 1.0) / self.xdif
        elif (self.ydif > self.xdif):
            self.xscale = (self.xdif * 1.0) / self.ydif

        self.myxmin = self.xmin()
        self.myxmax = self.xmax()
        self.myymin = self.ymin()
        self.myymax = self.ymax()
        
        for stroke in self.strokes:
            stroke.scale(self.myxmin, self.myxmax, self.myymin, self.myymax, self.xscale, self.yscale)

    # Given a class, this produces lines for an lg file.
    def lgline(self, clss):
        self.line = 'O, ' + self.ident + ', ' + clss + ', 1.0, ' + (', '.join(list(map((lambda s: str(s.ident)), self.strokes)))) + '\n'
        #do we need a newline here? Return to this if so.        
        return self.line
            
    def __str__(self):
        self.strng = 'Symbol'
        if self.correctClass != '':
            self.strng = self.strng + ' of class ' + self.correctClass
        self.strng = self.strng + ':\n Strokes:'
        for stroke in self.strokes:
            self.strng = self.strng + '\n' + str(stroke)
        return self.strng
    


# Holds the symbols from an inkml file.
class Expression:

    def __init__(self, name, symbols, relations):
        self.name = name
        self.symbols = symbols
        self.relations = relations
        self.classes = []


    def writeLG (self, directory, clss = None): #this _will_ probably break on windows style file paths.
        #Is there something cleaner from the libraries we can use?
        if directory[len(directory) -1] == '/':
            self.filename = directory + '/' + self.name + '.lg'
        else:
            self.filename = directory + self.name + '.lg'
        if (clss == None):
            print ("none clss")
            assert (len (list(self.classes)) == len (list(self.symbols)))
            self.clss = list(self.classes)
        else:
            self.clss = list(clss)
            
        self.symblines =  []
        self.i = 0

        #for c in (self.clss):
        #    print (c)
       # print (len(self.clss ), " ", len(list(self.symbols)))
        #print (self.clss)
        #for c in (self.clss):
            #print ( c)
        for symbol in self.symbols:
           # print (self.i)
            self.symblines.append(symbol.lgline(self.clss[self.i]))
            self.i = self.i + 1

        with (open (self.filename, 'w')) as f:
            
            for line in self.symblines:
                f.write(line)

            f.write('\n#Relations imported from original\n')
            
            for relation in self.relations:
                f.write(relation)

defaultClasses = ['!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']

# This stuff is used for reading strokes and symbols from files.


def readStroke(root, strokeNum):
    strokeElem = root.find("./{http://www.w3.org/2003/InkML}trace[@id='" + repr(strokeNum) + "']")
    strokeText = strokeElem.text.strip()
    pointStrings = strokeText.split(',')
    points = list(map( (lambda s: [float(n) for n in (s.strip()).split(' ')]), pointStrings))
    return Stroke(points, flip=True, ident=strokeNum)

#Are there any other substitutions of this type we need to make? Come back to this.
def doTruthSubs(text):
    if text == ',':
        return 'COMMA'
    else: 
        return text

def readSymbol(root, tracegroup):
    truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    identAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotationXML")    
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    assert( len(strokeElems) != 0)
    strokeNums = list(map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems)) #ensure that all these are really ints if we have trouble.
    strokes = list(map( (lambda n: readStroke(root, n)), strokeNums))
    if (truthAnnot == None):
        truthText = None
    else:
        truthText = doTruthSubs(truthAnnot.text)
    if identAnnot == None:
            #what do we even do with this?
            #messing with lg files depends on it.
            #for the momment, give it a bogus name and continue.
        idnt = str(strokeNums).replace(' ', '_')
    else:
        idnt = identAnnot.attrib['href'].replace(',', 'COMMA')
    return Symbol(strokes, correctClass=truthText, norm=True, ident=idnt )
    
    
def readFile(filename, warn=False):
    try:
        #print (filename)
        tree = ET.parse(filename)
        root = tree.getroot()
        tracegroups = root.findall('./*/{http://www.w3.org/2003/InkML}traceGroup')
        symbols = list(map((lambda t: readSymbol(root, t)), tracegroups))
        return symbols
    except:
        if warn:
            print("warning: unparsable file.")
        return []

def fnametolg(filename, lgdir):
    fdir, fname = os.path.split(filename)
    name, ext = os.path.splitext(fname)
    return os.path.join(lgdir, (name + ".lg"))


# this returns an expression class rather than just a list of symbols.
def readInkml(filename, lgdir, warn=False):
    symbols = readFile(filename, warn)
    rdir, filenm = os.path.split(filename)
    name, ext = os.path.splitext(filenm)
    lgfile = fnametolg(filename, lgdir)


    return Expression(name, symbols, readLG(lgfile))
    

def readLG(filename):

    with open(filename) as f:
        lines = f.readlines()

    relations = []
    for line in lines:
        if (line[0] == 'R' or line[0:2] =='EO'):
            relations.append(line)

    return relations        


def filenames(filename):
    inkmlre = re.compile('\.inkml$')
    fnames = []
    if(os.path.isdir(filename)):
        for root, dirs, files in os.walk(filename):
            for name in files:
                if(inkmlre.search(name) != None):
                    fnames.append(os.path.join(root, name))
    elif(inkmlre.search(filename) != None):
        fnames.append(filename)
    return fnames

def filepairs(filename, lgdir):
    fnames = filenames(filename)
    return list(map ((lambda f: (f, fnametolg(f, lgdir))), fnames))

def readDirectory(filename, warn=False):
    fnames = filenames(filename)
    return reduce( (lambda a, b : a + b), (list(map ((lambda f: readFile(f, warn)), fnames))), [])

def readInkmlDirectory(filename, lgdir, warn=False):
    fnames = filenames(filename)
    return list(map((lambda f: readInkml(f, lgdir, warn)), fnames))

def allSymbols(inkmls):
    return reduce( (lambda a, b: a + b), (list(map ((lambda i: i.symbols), inkmls))))

def symbsByClass(symbols):
    classes = {}
    for key in defaultClasses:
        classes[key] = []
    for symbol in symbols:
#        print (symbol)
        key = symbol.correctClass
        if (key not in classes):
            classes[key] = []
        classes[key].append(symbol)
    return classes

def symbClasses(symbols):
    keys = list(symbsByClass(symbols).keys())
    keys.sort()
    return keys

def exprClasses(inkmls):
    return symbClasses(allSymbols(inkmls))

def classNumbers(symbols, keys=None):
    if (keys == None):
        keys = list(symbsByClass(symbols).keys())
        keys.sort()
    cns = []
    for symbol in symbols:
       ct =  symbol.correctClass
       if ct==None:
           #cns.append(None)
           return None
       else:
           cns.append(keys.index(ct))
    return cns
    #return list(map((lambda symbol: keys.index(symbol.correctClass)), symbols))

#The function this is being fed to normalizes, so it doesn't matter that
#they don't sum to one.
def symbsPDF (symbols, keys=defaultClasses):
    if len(symbols) > 0:
        if isinstance(symbols[0], Expression):
            symbs = allSymbols(symbols)
        else:
            symbs = symbols
    

        clss = symbsByClass(symbs)
        counts = NP.array([len(clss[key]) for key in keys])
        return counts
    else:
        return numpy.zeros(len(keys))

def cleverSplit(fpairs, perc = (2.0/3), maxit = 100000):
    print ("reading files.")
    symbs = symbsByFPair(fpairs)


    print("constructing initial split")
    train, test = randSplit(fpairs, perc)

    print("getting initial PDFs")
    trnsymbs = NP.concatenate([symbs[t] for t in train])
    tstsymbs = NP.concatenate([symbs[t] for t in test])
    
    trainpdf = symbsPDF(trnsymbs)
    testpdf = symbsPDF(tstsymbs)
    #return (trainpdf, testpdf)
    #assert(len(trainpdf) == len(testpdf))
    print("getting initial entropy")
    entropy = scipy.stats.entropy(trainpdf, testpdf)
    print (entropy)
    count = 0
    while (count < maxit):
        #print("looping")
        i1 = random.randint(0, len(train)-1)
        i2 = random.randint(0, len(test)-1)
        
        i1_p = symbsPDF(symbs[train[i1]])
        i2_p = symbsPDF(symbs[train[i2]])
        new_trainpdf = (trainpdf - i1_p) + i2_p
        new_testpdf = (testpdf - i2_p) + i1_p

        new_entropy = scipy.stats.entropy(new_trainpdf, new_testpdf)
        if (new_entropy < entropy):
            print(new_entropy , " < " , entropy, ": swaping")
            testtmp = test[i2]
            test[i2] = train[i1]
            train[i1] = testtmp
            entropy = new_entropy
            trainpdf = new_trainpdf
            testpdf = new_testpdf

        count = count + 1

    print("split entropy: ", entropy)
    return (train, test)
    

def randSplit(items, perc = (2.0/3)):
    my_items = list(items)
    random.shuffle(my_items)
    splitnum = int(round(len(my_items) * perc))
    return (my_items[:splitnum], my_items[splitnum:])


def splitSymbols(symbols, trainPerc):
    classes = symbsByClass(symbols)
    training = []
    testing = []
    trainTarget = int(round(len(symbols) * trainPerc))
    testTarget = len(symbols) - trainTarget
    for clss, symbs in list(classes.items()):
        #consider dealing with unclassified symbols here if it is a problem.
        nsymbs = len(symbs)
        trainNum = int(round(nsymbs * trainPerc))
        random.shuffle(symbs)
        training = training + symbs[:trainNum]
        testing = testing + symbs[trainNum:]

    # Good enough unless the prof says otherwise.
    return( (training, testing))

#splits on a per-expression basis instead.
def splitExpressions(expressions, trainPerc):
    training = []
    testing = []
    exprs = expressions
    random.shuffle(exprs)
    trainNum = int(round (len(exprs) * trainPerc))
    training = training + exprs[:trainNum]
    testing = testing + exprs[trainNum:]

    #fancy stuff to ensure a good split goes here.
    #will try and add that today.

    return( (training, testing))

def symbsByFPair(fls):
    es = {}
    for fp in fls:
        #print fp 
        es[fp] = readFile(fp[0])
    return es

def splitFiles(inkmldir, lgdir, traindir, testdir, trainlg, testlg, trainPerc = (2.0 / 3.0)):

    training, testing = cleverSplit(list(filepairs(inkmldir, lgdir)))

   # fls = list(filepairs(inkmldir, lgdir))
   # random.shuffle(fls)
   # trainNum = int(round (len(fls) * trainPerc))
   # training = training + fls[:trainNum]
   # testing = testing + fls[trainNum:]

    for fpair in training:
        shutil.copy(fpair[0], traindir)
        shutil.copy(fpair[1], trainlg)

    for fpair in testing:
        shutil.copy(fpair[0], testdir)
        shutil.copy(fpair[1], testlg)
    #fancy stuff to ensure a good split goes here.
    #will try and add that today.

    return( (training, testing))

    
def pickleSymbols(symbols, filename):
    with open(filename, 'wb') as f:
        pickle.dump(symbols, f, pickle.HIGHEST_PROTOCOL)
        #note that this may cause problems if you try to unpickle with an older version.

def unpickleSymbols(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Normalize the data such that x or y -> (0,99) and maintain the aspect ratio
def normalize(symbols,scale):
    k=0
    for symbol in symbols:
        xmin = symbol.xmin()
        ymin = symbol.ymin()
        for i in range(len(symbol.strokes)):
            for j in range(len(symbol.strokes[i].xs)):
                symbol.strokes[i].xs[j] = (symbol.strokes[i].xs[j]-xmin)*scale/2
                symbol.strokes[i].ys[j] = (symbol.strokes[i].ys[j]-ymin)*scale/2
        symbols[k] = symbol
        k+=1    
    return(symbols)
