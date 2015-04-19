import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
import itertools
from functools import reduce
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
        return functools.reduce( (lambda a, b : a.union(b)), identSetList(), {})



def readTrueStrokeGroup(root, tracegroup):
    #truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    #identAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotationXML")    
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    assert( len(strokeElems) != 0)
    strokeNums = list(map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems)) #ensure that all these are really ints if we have trouble.
    strokes = list(map( (lambda n: readStroke(root, n)), strokeNums))
    #if (truthAnnot == None):
    #    truthText = None
    #else:
    #    truthText = doTruthSubs(truthAnnot.text)
    #if identAnnot == None:
            #what do we even do with this?
            #messing with lg files depends on it.
            #for the momment, give it a bogus name and continue.
    #    idnt = str(strokeNums).replace(', ', '_')
    #else:
    #    idnt = identAnnot.attrib['href'].replace(',', 'COMMA')
    return StrokeGroup(strokes, correctClass=truthText, norm=True, ident=idnt )
    



def readTrueStrokeGroup(root, tracegroup):
    #truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    #identAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotationXML")    
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    assert( len(strokeElems) != 0)
    strokeNums = list(map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems)) #ensure that all these are really ints if we have trouble.
    strokes = list(map( (lambda n: readStroke(root, n)), strokeNums))
    #if (truthAnnot == None):
    #    truthText = None
    #else:
    #    truthText = doTruthSubs(truthAnnot.text)
    #if identAnnot == None:
            #what do we even do with this?
            #messing with lg files depends on it.
            #for the momment, give it a bogus name and continue.
    #    idnt = str(strokeNums).replace(', ', '_')
    #else:
    #    idnt = identAnnot.attrib['href'].replace(',', 'COMMA')
    return StrokeGroup(strokes, correctClass=truthText, norm=True, ident=idnt )
