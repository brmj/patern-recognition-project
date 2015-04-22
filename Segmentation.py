import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
import itertools
import functools
import SymbolData



class StrokeGroup:
    """ Holds a set of strokes with no normalization or scaling. Used in segmentation. """
    def __init__(self, strokes, intersections = None):
        self.strokes = strokes
        if intersections is None:
            self.intersections = self.calcIntersections()
        else:
            self.intesections = intersections
            
    def toSymbol(self, correctClass = None, norm = True, ident = None):
        return (Symbol(self.strokes, correctClass, norm, ident))

    def strokeIdents(self):
        return set(map((lambda s: s.ident),self.strokes)) 

    def plot(self, show = True, clear = True):
        if clear:
            PLT.clf()
        for stroke in self.strokes:
            stroke.plot(show = False, clear = False)
        if show:
            PLT.show()

    def merge(self, other, inPlace = True):
        self.newInts = set(self.intersections).union(set(other.intersections)).union(set(self.calcIntersections(other)))
        if inPlace:
            self.strokes = self.strokes + other.strokes
            self.intersections = self.newInts
            return self
        else:
            return StrokeGroup(self.strokes + other.strokes, intersections = self.newInts)
        
    def minDist(self):
        return (NP.array(list(map((lambda s: s.minDist()), self.strokes))).min())

    def dists(self):
        return (list(map((lambda s: s.distances()), self.strokes)))

    def lens(self):
        return (list(map(NP.sum, self.dists())))
    
    def resample(self):
        self.md = self.minDist()
        map ((lambda s: s.resample(self.md)), self.strokes)
        
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

    def intersects(self, other):
        self.strokePairs = []
        for stroke1 in self.strokes:
            for stroke2 in other.strokes:
                self.strokePairs.append([stroke1, stroke2])
        for pair in strokePairs:
            if(stroke1.intersects(stroke2)):
                return True
            
        return False

    def intersections(self, other = None):
        self.newints = []
        self.strokePairs = []
        if (not other is None):
            for stroke1 in self.strokes:
                for stroke2 in other.strokes:
                    self.strokePairs.append([stroke1, stroke2])
        else:
            self.l = len(self.strokes)
            for i in range(0, l - 1):
                self.stroke1 = self.strokes[i]
                for j in range(i + 1, l):
                    self.stroke2 = self.strokes[j]
                    self.strokepairs.append([stroke1, stroke2])

        for pair in strokePairs:
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
        return list(map((lambda sg: sg.strokeIdents()), self.strokeGroups))

    def strokeIdents(self):
        return functools.reduce( (lambda a, b : a.union(b)), self.identSetList(), set())

    def inGroup(self, ident):
        for sg in self.identSetList():
            if ident in sg:
                return sg
        return None

    def mergeIntersecting(self):
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
                    
    
def comparePartitions(part1, part2, warn = False):
    correct = 0
    idents = part1.strokeIdents()
    if (warn and idents != part2.strokeIdents()):
        print ("Warning: strokes mismatch.")
    total = len (idents)

    for stroke in idents:
        group1 = part1.inGroup(stroke)
        group2 = part2.inGroup(stroke)
        assert (not group2 is None)
        if group1 == group2:
            correct +=1

    return [correct, total]

def comparePartitionsLists(l1, l2, warn = False):
    results = numpy.array(list(map((lambda ps: comparePartitions(ps[0], ps[1], warn)), zip(part1, part2))))
    sums = results.sum(axis = 0)
    perc = sums[0] / sums[1]
    return perc


def stupid_partition(strokes, name = None, relations = None):
    sgs = list(map((lambda s: StrokeGroup([s])), strokes))
    return Partition(sgs, name, relations)
    
def intersection_partition(strokes, name = None, relations = None):
    part = stupid_partition(strokes, name, relations)
    part.mergeIntersecting()
    return part

def mergePaired(mergefun, part):
    if len(part.strokeGroups) > 1:
            newGroups = []
            i = 1
            current = part.strokeGroups[0]
            while i < len(part.strokeGroups):
                #for the moment, go with a simplistic greedy approach.
                if mergefun(current, part.strokeGroups[i]):
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
            if mergefun(current, part.strokeGroups[i-1], part.strokeGroups[i]):
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
