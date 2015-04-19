import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
import itertools
import functools
import SymbolData



class StrokeGroup:
    """ Holds a set of strokes with no normalization or scaling. Used in segmentation. """
    def __init__(self, strokes):
        self.strokes = strokes

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


def stupid_partition(strokes, name = None, relations = None):
    sgs = list(map((lambda s: StrokeGroup([s])), strokes))
    return Partition(sgs, name, relations)
    

