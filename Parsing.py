import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
from functools import reduce
import SymbolData
import Segmentation



class SymbolNode:

    def __init__(self, strokeGroup, sns, sgs):
        self.sgIdent = strokeGroup.ident
        self.sns = sns
        self.sgs = sgs

        self.right = None
        self.above = None
        self.below = None
        self.inside = None
        self.subscript = None
        self.superscript = None


    def setrel(self, r, i):
        if r == 'R' or r == 'Right':
            self.right = i
        elif r == 'A' or r == 'Above':
            self.above = i
        elif r == 'B' or r == 'Below':
            self.below = i
        elif r == 'I' or r == 'Inside':
            self.inside = i
        elif r == 'Sub' or r == 'Subscript':
            self.subscript = i
        elif r == 'Sup' or r == 'Superscript':
            self.superscript = i
        else:
            print ("Bad rel: ", r)

            
    def sg(self):
        return sgs[sgIdent]

    def rightmost(self):
        if self.right == None:
            return sgIdent
        else:
            return self.sns[self.right].rightmost()


    def isChild(self, sg):
        if (self.right == sg or
            self.above == sg or
            self.below == sg or
            self.inside == sg or
            self.subscript == sg or
            self.superscript == sg):
            return True
        else:
            if (self.right != None):
                if self.sns[self.right].isChild(sg):
                    return True
            elif (self.above != None):
                if self.sns[self.above].isChild(sg):
                    return True
            elif (self.below != None):
                if self.sns[self.below].isChild(sg):
                    return True
            elif (self.inside != None):
                if self.sns[self.inside].isChild(sg):
                    return True
            elif (self.subscript != None):
                if self.sns[self.subscript].isChild(sg):
                    return True
            elif (self.superscript != None):
                if self.sns[self.superscript].isChild(sg):
                    return True
            else:
                return False

    def lg_rel_lines(self):
        return self.lg_rel_lines_EO()
            
    def lg_rel_lines_EO(self):
        self.l1 = []
        self.l2 = []
        self.lineprefix = "EO, " + self.sgIdent + ", "
        if (self.right != None):
            self.l1.append (self.lineprefix + self.sns[self.right].sgIdent + ", Right, 1.0")
            self.l2 = self.l2 +  (self.sns[self.right].lg_rel_lines_EO())
        if (self.above != None):
            self.l1.append (self.lineprefix + self.sns[self.above].sgIdent + ", Above, 1.0")
            self.l2 = self.l2 + (self.sns[self.above].lg_rel_lines_EO())
        if (self.below != None):
            self.l1.append (self.lineprefix + self.sns[self.below].sgIdent + ", Below, 1.0")
            self.l2 = self.l2 + (self.sns[self.below].lg_rel_lines_EO())
        if (self.inside != None):
            self.l1.append (self.lineprefix + self.sns[self.inside].sgIdent + ", Inside, 1.0")
            self.l2 = self.l2 + (self.sns[self.inside].lg_rel_lines_EO())
        if (self.subscript != None):
            self.l1.append (self.lineprefix + self.sns[self.subscript].sgIdent + ", Sub, 1.0")
            self.l2 = self.l2 + (self.sns[self.subscript].lg_rel_lines_EO())
        if (self.superscript != None):
            self.l1.append (self.lineprefix + self.sns[self.superscript].sgIdent + ", Sup, 1.0")
            self.l2 = self.l2 + (self.sns[self.superscript].lg_rel_lines_EO())

        return (self.l1 + self.l2)
            
    def lg_rel_lines_E(self):
        self.l1 = []
        self.l2 = []
        self.lineprefix = "E, " + self.sgIdent + ", "
        if (self.right != None):
            self.l1.append (self.lineprefix + self.sns[self.right].sgIdent + ", R")
            self.l2 = self.l2 +  (self.sns[self.right].lg_rel_lines_E())
        if (self.above != None):
            self.l1.append (self.lineprefix + self.sns[self.above].sgIdent + ", A")
            self.l2 = self.l2 +  (self.sns[self.above].lg_rel_lines_E())
        if (self.below != None):
            self.l1.append (self.lineprefix + self.sns[self.below].sgIdent + ", B")
            self.l2 = self.l2 + (self.sns[self.below].lg_rel_lines_E())
        if (self.inside != None):
            self.l1.append (self.lineprefix + self.sns[self.inside].sgIdent + ", I")
            self.l2 = self.l2 + (self.sns[self.inside].lg_rel_lines_E())
        if (self.subscript != None):
            self.l1.append (self.lineprefix + self.sns[self.subscript].sgIdent + ", SUB")
            self.l2 = self.l2 + (self.sns[self.subscript].lg_rel_lines_E())
        if (self.superscript != None):
            self.l1.append (self.lineprefix + self.sns[self.superscript].sgIdent + ", SUP")
            self.l2 = self.l2 + (self.sns[self.superscript].lg_rel_lines_E())

        return (self.l1 + self.l2)


class Parse:
    def __init__(self, sgs = {}):
        self.head = None
        self.sgs = sgs
        self.sns = {}

    def lg_rel_lines(self):
        if self.head == None:
            return []
        else:
            return self.sns[self.head].lg_rel_lines()
        
def leftToRight(strokeGroups, sgs = None):
    if sgs == None: #then they really are just stroke groups
        return sorted(strokeGroups, key=(lambda sg: sg.xmin()))
    else: #otherwise, they are idents.
        return sorted(strokeGroups, key=(lambda sg: sgs[sg].xmin()))

        
def veryStupidParse(partition): #Assumes the stroke groups are left to right, in order. Not a smart assumption. Just a baseline/testing sort of thing.
    sgs = {}
    for sg in partition.strokeGroups:
        sgs[sg.ident] = sg

    sns = {}
    parse = Parse(sgs)
    if len (sgs) == 0:
        parse.head = None
    else:
        parse.head = partition.strokeGroups[0].ident
        sns[parse.head] = SymbolNode(partition.strokeGroups[0], sns, sgs) #pretend these are pointers to get how it works.
        
        prev = partition.strokeGroups[0].ident
        for sg in partition.strokeGroups[1:]:
            sns[prev].right = sg.ident
            sns[sg.ident] = SymbolNode(sg, sns, sgs)
            prev = sg.ident

    parse.sns = sns
    return parse

def lessStupidParse(partition): #Assumes everything is in a single line. Not a smart assumption. Just a baseline/testing sort of thing.
    sgs = {}
    for sg in partition.strokeGroups:
        sgs[sg.ident] = sg

    sns = {}
    parse = Parse(sgs)
    l2r = leftToRight(list(map ((lambda sg: sg.ident), partition.strokeGroups)), sgs)
    if len (sgs) == 0:
        parse.head = None
    else:
        parse.head = l2r[0]
        sns[parse.head] = SymbolNode(sgs[l2r[0]], sns, sgs) #pretend these are pointers to get how it works.
        
        prev = l2r[0]
        for sgi in l2r[1:]:
            sns[prev].right = sgi
            sns[sgi] = SymbolNode(sgs[sgi], sns, sgs)
            prev = sgi

    parse.sns = sns
    return parse
        
def trueParse(partition, rels = None): #MUST be a true segmentation with ground truth idents and matching relationship lines from an lg file.
    if rels is None:
        rels = partition.relations
    sgs = {}
    sns = {}
    for sg in partition.strokeGroups:
        sgs[sg.ident] = sg
        sns[sg.ident] = SymbolNode(sg, sns, sgs)
        
    parse = Parse(sgs)
    prels = map(parseRel, rels)
    for pr in prels:
        sns[pr[0]].setrel(pr[2], pr[1]) #adds the appropriate relationship.

    parse.head = getHead(sns, sgs)

    parse.sns = sns
    return parse


def parseRel(rel):
    splt = rel.split(', ')
    return tuple(splt[1:4])

    
def getHead(sns, sgs):
    idents = list(sns.keys())
    parentless = set(idents)
    parented = set([])

    def procChild(id):
        if not id is None:
            if parented.intersection({id}) == {id}:
                print ("ERROR: not a tree!")
            elif parentless.intersection({id}) == {id}:
                parentless.remove(id)
                parented.add(id)
            else:
                print(id , " not anticipated.")

            
    for si in idents:
        sn = sns[si]
        procChild(sn.right)
        procChild(sn.above)
        procChild(sn.below)
        procChild(sn.inside)
        procChild(sn.subscript)
        procChild(sn.superscript)

    if len(parentless) != 1:
        print ("Warning: multiple trees.")
        print ("roots: " , parentless)
        
    return parentless.pop()
            
            
