import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
from pylab import *



""" Contains representations for the relevant data,
    As well as functions for reading and processing it. """

class Stroke:
    """Represents a stroke as an n by 2 matrix, with the rows of
      the matrix equivelent to points from first to last. """
    def __init__(self, points):
        self.xs = []
        self.ys = []
        for point in points:
            self.addPoint(point)

    def plot(self, show = True, clear = True):
        if clear:
           PLT.clf()
        PLT.plot(self.xs, self.ys, 'ko-' )
           
        if show:
            PLT.show()

    def addPoint(self, point):
        self.xs.append(point[0])
        self.ys.append(point[1])

    def asPoints(self):
        return (zip(self.xs, self.ys))

    def scale(self, xmin, xmax, ymin, ymax):
        self.xs = map( (lambda x: ((x - xmin) * 1.0 / (xmax - xmin))), self.xs)
        self.ys = map( (lambda y: ((y - ymin) * 1.0 / (ymax - ymin))), self.ys)

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
    def __init__(self, strokes, correctClass = '', norm = True):
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

    def normalize(self):
        self.xmin = min(map( (lambda stroke: stroke.xmin()), self.strokes))
        self.xmax = max(map( (lambda stroke: stroke.xmax()), self.strokes))
        self.ymin = min(map( (lambda stroke: stroke.ymin()), self.strokes))
        self.ymax = max(map( (lambda stroke: stroke.ymax()), self.strokes))

        for stroke in self.strokes:
            stroke.scale(self.xmin, self.xmax, self.ymin, self.ymax)
            
    def __str__(self):
        self.strng = 'Symbol'
        if self.correctClass != '':
            self.strng = self.strng + ' of class ' + self.correctClass
        self.strng = self.strng + ':\n Strokes:'
        for stroke in self.strokes:
            self.strng = self.strng + '\n' + str(stroke)
        return self.strng
    
