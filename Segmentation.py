
import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
import itertools
import functools
import os
import SymbolData
import Classification
import SegFeatures
from functools import reduce



class StrokeGroup:
    """ Holds a set of strokes with no normalization or scaling. Used in segmentation. """
    def __init__(self, strokes, intersections = None, correctClass = None, norm = True, ident = None):
        self.strokes = strokes
        assert(strokes != [] and not strokes is None)
        if intersections is None:
            self.intersections = self.calcIntersections()
        else:
            self.intersections = intersections

        self.correctClass = correctClass
        self.norm = norm
        self.ident = ident

        #let's see how it goes... Should still add stuff to deal with _almost_ colinear points, _almost_ identical, etc.
        #self.resample()
        # self.strokenum = len(self.strokes)
        #if(self.norm):
        #self.uniformResample(28) #a good start, since the prof suggested 30.
            
    def toSymbol(self):
        self.newstrokes = self.strokes
        return (SymbolData.Symbol(self.newstrokes, self.correctClass, self.norm, self.ident, self.intersections, self.strokenum))

    def strokeIdents(self):
        #print("sg ", set(map((lambda s: s.ident),self.strokes)) )
        return set(map((lambda s: s.ident),self.strokes)) 

    def plot(self, show = True, clear = True):
        if clear:
            PLT.clf()
        for stroke in self.strokes:
            stroke.plot(show = False, clear = False)
        if show:
            PLT.show()

    def copy(self):
        return StrokeGroup(list(map ((lambda s: s.copy()), self.strokes)), intersections = self.intersections, correctClass = self.correctClass, norm = self.norm, ident = self.ident)
            
    def merge(self, other, inPlace = True):
        self.newInts = set(self.intersections).union(set(other.intersections)).union(set(self.calcIntersections(other)))
        if inPlace:
            self.strokes = self.strokes + other.strokes
            self.intersections = self.newInts
            return self
        else:
            return StrokeGroup(self.strokes + other.strokes, intersections = self.newInts)

    def uniformResample(self, divs):
        self.newident =  str(list(map((lambda s: s.ident), self.strokes)))
        self.newstroke = SymbolData.Stroke(reduce ((lambda a, b: a + b.asPoints()), self.strokes, []), ident = self.newident)
        self.newstroke.uniformResample(divs)
        self.strokes = [self.newstroke]
        
    def minDist(self):
        return (NP.array(list(map((lambda s: s.minDist()), self.strokes))).min())

    def dists(self):
        return (list(map((lambda s: s.distances()), self.strokes)))

    def lens(self):
        return (list(map((lambda s: s.totlen()), self.strokes)))

    def strokenum(self):
        return len(self.strokes)

    def totlen(self):
        return NP.sum(self.lens())
    
    def resample(self):
        self.md = self.minDist()
        return (list(map((lambda s: s.resample(self.md)), self.strokes)))
        
    def xmin(self):
        return min(list(map( (lambda stroke: stroke.xmin()), self.strokes)))

    def xmax(self):
        return max(list(map( (lambda stroke: stroke.xmax()), self.strokes)))

    def ymin(self):
        return min(list(map( (lambda stroke: stroke.ymin()), self.strokes)))

    def ymax(self):
        return max(list(map( (lambda stroke: stroke.ymax()), self.strokes)))

    def points(self):
        return reduce( (lambda a, b : a + b), (list(map ((lambda f: f.asPoints()), self.strokes))), [])

    def xs(self):
        return reduce( (lambda a, b : a + b), (list(map ((lambda f: f.xs), self.strokes))), [])

    def ys(self):
        return reduce( (lambda a, b : a + b), (list(map ((lambda f: f.ys), self.strokes))), [])

    def intersects(self, other):
        self.strokePairs = []
        for stroke1 in self.strokes:
            for stroke2 in other.strokes:
                self.strokePairs.append([stroke1, stroke2])
        for pair in self.strokePairs:
            if(stroke1.intersects(stroke2)):
                return True
            
        return False

    def calcIntersections(self, other = None):
        self.newints = []
        self.strokePairs = []
        if (not other is None):
            for stroke1 in self.strokes:
                for stroke2 in other.strokes:
                    self.strokePairs.append((stroke1, stroke2))
        else:
            self.l = len(self.strokes)
            for i in range(0, self.l - 1):
                self.stroke1 = self.strokes[i]
                for j in range(i + 1, self.l):
                    self.stroke2 = self.strokes[j]
                    self.strokePairs.append((self.stroke1, self.stroke2))

        for (stroke1, stroke2) in self.strokePairs:
            self.newints = self.newints + stroke1.intersections(stroke2)
        return self.newints
             
    def __str__(self):
        return ("StrokeGroup " + str(list(map((lambda s: s.ident), self.strokes))))
    

class Partition:
    """A partition of the strokes in a file into segments. This is to expressions what a stroke group is to symbols, bassically."""
    def __init__(self, strokeGroups, name = None, relations = None):
        self.strokeGroups = strokeGroups
        self.name = name
        self.relations = relations

    def toExpression(self):
        return Expression(self.name, list(map((lambda sg: sg.toSymbol()), self.strokeGroups)), self.relations)

    def identSetList(self):
        #print ("Part: ", list(map((lambda sg: sg.strokeIdents()), self.strokeGroups)))
        return list(map((lambda sg: sg.strokeIdents()), self.strokeGroups))

    def byIdentSet(self, iset):
        self.isetl = self.identSetList()
        if iset in self.isetl:
            return self.strokeGroups[ self.isetl.index(iset) ]
        else:
            return None
    
    def strokeIdents(self):
        return reduce( (lambda a, b : a.union(b)), self.identSetList(), set())

    def toSymbols(self):
        return list(map((lambda sg: sg.toSymbol()), self.strokeGroups))
    
    def inGroup(self, ident):
        for sg in self.identSetList():
            if ident in sg:
                return sg
        return None

    def mergeIntersecting(self):
        self.idents = self.strokeIdents()
        if len(self.strokeGroups) > 1:
            self.newGroups = []
            self.i = 1
            self.current = self.strokeGroups[0]
            while self.i < len(self.strokeGroups):
                if self.current.intersects( self.strokeGroups[self.i]):
                    self.current = self.current.merge(self.strokeGroups[self.i])
                else:
                    self.newGroups.append(self.current)
                    self.current = self.strokeGroups[self.i]
                self.i+=1
            self.newGroups.append(self.current)
            self.strokeGroups = self.newGroups
        assert (self.strokeIdents() == self.idents)
        

def readAndSegment(filename, segfun, warn = False, raw = False):
    strokes = SymbolData.readFileStrokes(filename, warn)
    partition = segfun(strokes)
    name, ext = os.path.splitext(filename)
    if raw:
        return partition
    else:
        return SymbolData.Expression(name, partition.toSymbols(), None)
        
def readAndSegmentDirectory(filename, segfun, warn=False, raw = False):
    fnames = SymbolData.filenames(filename)
    return list(map((lambda f: readAndSegment(f, segfun, warn, raw)), fnames))
            
def comparePartitions(part1, part2, warn = False):
    correct = 0
    idents = part1.strokeIdents()
    if (warn and idents != part2.strokeIdents()):
        print ("Warning: strokes mismatch.")
    total = len (idents)
    #print (idents)
    #print (total)
    #print (part2.strokeGroups)
    for stroke in idents:
        group1 = part1.inGroup(stroke)
        group2 = part2.inGroup(stroke)
        #assert (not group2 is None)
        if group1 == group2:
            correct +=1

    return [correct, total]



def comparePartitionsLists(l1, l2, warn = False):
    results = NP.array(list(map((lambda ps: comparePartitions(ps[0], ps[1], warn)), zip(l1, l2))))
    sums = results.sum(axis = 0)
    #print (sums)
    perc = sums[0] / sums[1]
    return perc

def testSegmentation(filename, lgdir, segfun, warn = False):
    exprs = SymbolData.readInkmlDirectory(filename, lgdir, warn)
    parts = readAndSegmentDirectory(filename, segfun, warn, raw = True)
    segrate = comparePartitionsLists(parts, exprs)
    print (segrate)
    return segrate

def pairIdentSet(pair):
    return pair[0].strokeIdents().union(pair[1].strokeIdents())

def lPairs(l):
    if (len(l) < 2):
        return []
    else:
        return list(zip(l[:len(l) -1], l[1:]))

def pairIdentSets(part):
    return list(map(pairIdentSet, lPairs(part.strokeGroups)))

def inTruth(part, truePart):
    return list(map((lambda s: not truePart.byIdentSet(s)  is None), pairIdentSets(part)))

def fileTrainData(filename):
    #specifically not indulging in premature optimization here...
    interPart = readAndSegment(filename, intersection_partition, raw = True)
    truePart = readTruePart(filename)

    pairs = lPairs(interPart.strokeGroups)
    features = SegFeatures.features(pairs)
    
    truths = inTruth(interPart, truePart)
    def boolToInt(b):
        if b:
            return 1
        else:
            return 0

    classes = list(map(boolToInt, truths))

    return list(zip(features, truths))
    

def readTrueSG(root, tracegroup):
    truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    identAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotationXML")    
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    assert( len(strokeElems) != 0)
    strokeNums = list(map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems)) #ensure that all these are really ints if we have trouble.
    strokes = list(map( (lambda n: SymbolData.readStroke(root, n)), strokeNums))
    if (truthAnnot == None):
        truthText = None
    else:
        truthText = SymbolData.doTruthSubs(truthAnnot.text)
    if identAnnot == None:
            #what do we even do with this?
            #messing with lg files depends on it.
            #for the momment, give it a bogus name and continue.
        idnt = str(strokeNums).replace(', ', '_')
    else:
        idnt = identAnnot.attrib['href'].replace(',', 'COMMA')

    sg = StrokeGroup(strokes, correctClass = truthText, norm=True, ident=idnt )
    return sg
    
def readTrueSGsFile(filename, warn=False):
    tree = None
    try:
        #print (filename)
        tree = ET.parse(filename)
    except:
        if warn:
            print("warning: unparsable file.")
        return []
    root = tree.getroot()
    tracegroups = root.findall('./*/{http://www.w3.org/2003/InkML}traceGroup')
    sgs = list(map((lambda t: readTrueSG(root, t)), tracegroups))
    return sgs

def readTrueSGsDirectory(filename, warn=False):
    fnames = SymbolData.filenames(filename)
    return reduce( (lambda a, b : a + b), (list(map ((lambda f: readTrueSGsFile(f, warn)), fnames))), [])

def readTruePart(filename, warn=False):
    sgs = readTrueSGsFile(filename, warn)
    return Partition(sgs)

def readTruePartsDirectory(filename, warn=False):
    fnames = SymbolData.filenames(filename)
    return list(map((lambda f: readTruePart(f, lgdir, warn)), fnames))

def stupid_partition(strokes, name = None, relations = None):
    sgs = list(map((lambda s: StrokeGroup([s])), strokes))
    return Partition(sgs, name, relations)
    
def intersection_partition(strokes, name = None, relations = None):
    part = stupid_partition(strokes, name, relations)
    part.mergeIntersecting()
    return part

def mkClassiffyPairMergeFun(model, pca, renormalize = True):
    def isTwoPart (cls):
        return cls in SymbolData.twoGroup
    def makesymb(sgs):
        return (reduce((lambda sg1, sg2: sg1.merge(sg2, inPlace=False)), list(map((lambda sg: sg.copy()),sgs)))).toSymbol()
    def twoPartSymb(symb):
        cls = Classification.classifySymbol(symb, model, pca, renormalize)
        return isTwoPart(cls)
    mergeFun = (lambda sgs: twoPartSymb(makesymb(sgs)))
    return mergeFun

def mkIsThreeGroupMergeFun(model, pca, renormalize = True):
    def isThreePart (clss):
        if (len(clss) != 3):
            return False
        if ((SymbolData.defaultClasses[clss[0]] == 's' or SymbolData.defaultClasses[clss[0]] == 'S') and
            (SymbolData.defaultClasses[clss[1]] == 'i' or SymbolData.defaultClasses[clss[1]] == 'I'
             or SymbolData.defaultClasses[clss[1]] == '|' or (SymbolData.defaultClasses[clss[1]] == '1')) and
            (SymbolData.defaultClasses[clss[2]] == 'n' or SymbolData.defaultClasses[clss[2]] == 'N')):
            return True
        if ((SymbolData.defaultClasses[clss[0]] == 'c' or SymbolData.defaultClasses[clss[0]] == 'C'
             or SymbolData.defaultClasses[clss[0]] == '(') and
            (SymbolData.defaultClasses[clss[1]] == 'o' or SymbolData.defaultClasses[clss[1]] == 'O'
             or SymbolData.defaultClasses[clss[1]] == '0') and
            (SymbolData.defaultClasses[clss[2]] == 's' or SymbolData.defaultClasses[clss[2]] == 'S')):
            return True
        if ((SymbolData.defaultClasses[clss[0]] == 't' or SymbolData.defaultClasses[clss[0]] == 'T'
             or (SymbolData.defaultClasses[clss[0]] == '+')) and
            (SymbolData.defaultClasses[clss[1]] == 'a' or SymbolData.defaultClasses[clss[1]] == 'A') and
            (SymbolData.defaultClasses[clss[2]] == 'n' or SymbolData.defaultClasses[clss[2]] == 'N')):
            return True
        if ((SymbolData.defaultClasses[clss[0]] == 'l' or SymbolData.defaultClasses[clss[0]] == 'L') and
            (SymbolData.defaultClasses[clss[1]] == 'i' or SymbolData.defaultClasses[clss[1]] == 'I'
             or SymbolData.defaultClasses[clss[1]] == '|' or (SymbolData.defaultClasses[clss[1]] == '1')) and
            (SymbolData.defaultClasses[clss[2]] == 'm' or SymbolData.defaultClasses[clss[2]] == 'M')):
            return True
        if ((SymbolData.defaultClasses[clss[0]] == 'd' or SymbolData.defaultClasses[clss[0]] == 'D') and
            (SymbolData.defaultClasses[clss[1]] == 'i' or SymbolData.defaultClasses[clss[1]] == 'I'
             or SymbolData.defaultClasses[clss[1]] == '|' or (SymbolData.defaultClasses[clss[1]] == '1')) and
            (SymbolData.defaultClasses[clss[2]] == 'V' or SymbolData.defaultClasses[clss[2]] == 'v')):
            return True
        if ((SymbolData.defaultClasses[clss[0]] == 'l' or SymbolData.defaultClasses[clss[0]] == 'L'
             or SymbolData.defaultClasses[clss[0]] == '(') and
            (SymbolData.defaultClasses[clss[1]] == 'o' or SymbolData.defaultClasses[clss[1]] == 'O'
             or SymbolData.defaultClasses[clss[1]] == '0') and
            (SymbolData.defaultClasses[clss[2]] == 'g' or SymbolData.defaultClasses[clss[2]] == 'G')):
            return True
        return False
    
    def makesymb(sgs):
        return (reduce((lambda sg1, sg2: sg1.merge(sg2, inPlace=False)), list(map((lambda sg: sg.copy()),sgs)))).toSymbol()
    def twoPartSymb(symb):
        clss = Classification.classifySymbol(symb, model, pca, renormalize)
        return isTwoPart(clss)
    mergeFun = (lambda sgs: isThreePart(makesymb(sgs)))
    return mergeFun

def mergePaired(mergefun, part):
    if len(part.strokeGroups) > 1:
            newGroups = []
            i = 1
            current = part.strokeGroups[0]
            while i < len(part.strokeGroups):
                #for the moment, go with a simplistic greedy approach.
                if mergefun([current, part.strokeGroups[i]]):
                    newGroups.append(current.merge(part.strokeGroups[i]))
                    if i +1 < len (part.strokeGroups):
                        current = part.strokeGroups [i + 1]
                    else:
                        current = None
                    i +=2
                else:
                    newGroups.append(current)
                    current = strokeGroups[i]
                    i+=1
            if (not current is None):
                newGroups.append(current)
            return Partition(newGroups, part.name, part.relations)
    else:
        return part


def mergeTripled(mergeFun, part):
    if len (part.strokeGroups) > 2:
        newGroups = []
        i = 2
        current = part.strokeGroups[0]
        while i < len(part.strokeGroups):
            #for the moment, go with a simplistic greedy approach.

            if mergefun([current, part.strokeGroups[i-1], part.strokeGroups[i]]):
                newGroups.append(current.merge(part.strokeGroups[i-1]).merge(part.strokeGroups[i]))
                if i +1 < len (part.strokeGroups):
                    current = part.strokeGroups [i + 1]
                else:
                    current = None
                    i +=3
            else:
                newGroups.append(current)
                current = strokeGroups[i-1]
                i+=1
        if (not current is None):
            newGroups.append(current)
            if (i-1 < len(part.strokeGroups)):
                newGroups.append(part.strokeGroups[i -1])
                
        return Partition(newGroups, part.name, part.relations)
    else:
        return part

def mkCleverPart(model, pca, renormalize = True, name = None, relations = None):
    def cleverPart(strokes):
        part = intersection_partition(strokes, name = None, relations = None)
        merge2fun = mkClassiffyPairMergeFun(model, pca, renormalize)
        part2 = mergePaired(merge2fun, part)
        merge3fun = mkIsThreeGroupMergeFun(model, pca, renormalize)
        part3 = mergeTripled(merge3fun, part2)
        return part3

    return (lambda strokes: cleverPart(strokes))
