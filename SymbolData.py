import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
from pylab import *
import os
import re
import random
import pickle
import functools


""" Contains representations for the relevant data,
    As well as functions for reading and processing it. """

class Stroke:
    """Represents a stroke as an n by 2 matrix, with the rows of
      the matrix equivelent to points from first to last. """
    def __init__(self, points, flip = False):
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
        return (zip(self.xs, self.ys))

    def scale(self, xmin, xmax, ymin, ymax, xscale, yscale):
        if (xmax != xmin):
            self.xs = map( (lambda x: xscale * ((x - xmin) * 1.0 / (xmax - xmin))), self.xs)
        else:
            self.xs = map( (lambda x: 0), self.xs)
        if (ymax != ymin):
            self.ys = map( (lambda y: yscale * ((y - ymin) * 1.0 / (ymax - ymin))), self.ys)
        else:
            self.ys = map( (lambda y: 0), self.ys)
        self.xs = map( (lambda x: (x * 2) - xscale), self.xs)
        self.ys = map( (lambda y: (y * 2) - yscale), self.ys)

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
    def __init__(self, strokes, correctClass = None, norm = True):
        self.strokes = strokes
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
        return min(map( (lambda stroke: stroke.xmin()), self.strokes))

    def xmax(self):
        return max(map( (lambda stroke: stroke.xmax()), self.strokes))

    def ymin(self):
        return min(map( (lambda stroke: stroke.ymin()), self.strokes))

    def ymax(self):
        return max(map( (lambda stroke: stroke.ymax()), self.strokes))

    def points(self):
        return functools.reduce( (lambda a, b : a + b), (map ((lambda f: f.asPoints()), self.strokes)), [])

    def xs(self):
        return functools.reduce( (lambda a, b : a + b), (map ((lambda f: f.xs), self.strokes)), [])

    def ys(self):
        return functools.reduce( (lambda a, b : a + b), (map ((lambda f: f.ys), self.strokes)), [])
    
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
            
    def __str__(self):
        self.strng = 'Symbol'
        if self.correctClass != '':
            self.strng = self.strng + ' of class ' + self.correctClass
        self.strng = self.strng + ':\n Strokes:'
        for stroke in self.strokes:
            self.strng = self.strng + '\n' + str(stroke)
        return self.strng
    




# This stuff is used for reading strokes and symbols from files.
# Code for doing a propper split will also have to go here, I think.


def readStroke(root, strokeNum):
    strokeElem = root.find("./{http://www.w3.org/2003/InkML}trace[@id='" + repr(strokeNum) + "']")
    strokeText = strokeElem.text.strip()
    pointStrings = strokeText.split(',')
    points = map( (lambda s: map(lambda n: float(n), (s.strip()).split(' '))), pointStrings)
    return Stroke(points, flip=True)

def readSymbol(root, tracegroup):
    truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    strokeNums = map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems) #ensure that all these are really ints if we have trouble.
    strokes = map( (lambda n: readStroke(root, n)), strokeNums)
    if (truthAnnot == None):
        return Symbol(strokes)
    else:
        return Symbol(strokes, correctClass=truthAnnot.text)
    
    
def readFile(filename, warn=False):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        tracegroups = root.findall('./*/{http://www.w3.org/2003/InkML}traceGroup')
        symbols = map((lambda t: readSymbol(root, t)), tracegroups)
        return symbols
    except:
        if warn:
            print("warning: unparsable file.")
        return []

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

def readDirectory(filename, warn=False):
    fnames = filenames(filename)
    return reduce( (lambda a, b : a + b), (map ((lambda f: readFile(f, warn)), fnames)), [])

def symbsByClass(symbols):
    classes = {}
    for symbol in symbols:
        key = symbol.correctClass
        if (not classes.has_key(key)):
            classes[key] = []
        classes[key].append(symbol)
    return classes

def classNumbers(symbols, keys=None):
    if (keys == None):
        keys = symbsByClass(symbols).keys()
    keys.sort()
    return map ((lambda symbol: keys.index(symbol.correctClass)), symbols)

def splitSymbols(symbols, trainPerc):
    classes = symbsByClass(symbols)
    training = []
    testing = []
    trainTarget = int(round(len(symbols) * trainPerc))
    testTarget = len(symbols) - trainTarget
    for clss, symbs in classes.items():
        #consider dealing with unclassified symbols here if it is a problem.
        nsymbs = len(symbs)
        trainNum = int(round(nsymbs * trainPerc))
        random.shuffle(symbs)
        training = training + symbs[:trainNum]
        testing = testing + symbs[trainNum:]

    # Good enough unless the prof says otherwise.
    return( (training, testing))

def pickleSymbols(symbols, filename):
    with open(filename, 'wb') as f:
        pickle.dump(symbols, f, pickle.HIGHEST_PROTOCOL)
        #note that this may cause problems if you try to unpickle with an older version.

def unpickleSymbols(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
