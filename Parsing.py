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

def roots(sgidents, sgs): #Watch very carefully to make sure classes end up loaded in properly once we are using the stuff we segment and parse. This could be bad.
    return list(filter((lambda sgi: sgs[sgi].correctClass == '\\sqrt'), sgidents))

def hLines(sgidents, sgs): #Watch very carefully to make sure classes end up loaded in properly once we are using the stuff we segment and parse. This could be bad.
    hls =  list(filter((lambda sgi: sgs[sgi].correctClass == '-' or sgs[sgi].correctClass == '\\sum' or sgs[sgi].correctClass == '\\int'), sgidents))
    return sorted(hls, key=(lambda sgi: sgs[sgi].xdist() * -1 )) #Sort them widest to narrowist, since that is the order we want them in.

def strictlyAbove(sgi, sgidents, sgs):
    xmin = sgs[sgi].xmin()
    xmax = sgs[sgi].xmax()
    ymax = sgs[sgi].ymax()
    return list(filter( (lambda sg: sgs[sg].xmin() >= xmin and sgs[sg].xmax() <= xmax and sgs[sg].ymin() > ymax), sgidents))

def strictlyBelow(sgi, sgidents, sgs):
    xmin = sgs[sgi].xmin()
    xmax = sgs[sgi].xmax()
    ymin = sgs[sgi].ymin()
    return list(filter( (lambda sg: sgs[sg].xmin() >= xmin and sgs[sg].xmax() <= xmax and sgs[sg].ymax() < ymin), sgidents))
    
def inRoot (sgi, sgidents, sgs): #Heurisitc for if things are in a root.
    inside = []
    above = [] #will have to see if using this helps. If not, tweak it or remove it I suppose.
    xmin = sgs[sgi].xmin()
    xmax = sgs[sgi].xmax()
    ymin = sgs[sgi].ymin()
    ymax = sgs[sgi].ymax()
    ydist = sgs[sgi].ydist()
    for sgid in sgidents: #see if the top left corner is in the bounding box. If so, it is probably inside the root.
        sg_xdist = sgs[sgid].xdist()
        sg_x = sgs[sgid].xmin()
        sg_y = sgs[sgid].ymax()

        if (sg_x > xmin and sg_x + 0.25 * sg_xdist < xmax and sg_y > ymin + 0.25 * ydist and sg_y < ymax):
            inside.append(sgid)
        else: #if the bottom right corner is in the bounding box and it is a three, it is probably Above the root, given that it isn't inside.
             sg_x = sgs[sgid].xmax()
             sg_y = sgs[sgid].ymin()
             if (sg_x > xmin and sg_x < xmax and sg_y > ymin and sg_y < ymax and sgs[sgid].correctClass == '3'):
                 above.append(sgid)

    return (inside, above)

    
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

def badParse(partition): #Should never be right. Testing thing.
    sgs = {}
    for sg in partition.strokeGroups:
        sgs[sg.ident] = sg

    sns = {}
    parse = Parse(sgs)
    l2r = leftToRight(list(map ((lambda sg: sg.ident), partition.strokeGroups)), sgs)
    r2l = list(reversed(l2r))
    if len (sgs) == 0:
        parse.head = None
    else:
        parse.head = r2l[0]
        sns[parse.head] = SymbolNode(sgs[r2l[0]], sns, sgs) #pretend these are pointers to get how it works.
        
        prev = r2l[0]
        for sgi in r2l[1:]:
            sns[prev].right = sgi
            sns[sgi] = SymbolNode(sgs[sgi], sns, sgs)
            prev = sgi

    parse.sns = sns
    return parse


def recursiveParse(partition): #Loop through a series of reductions like in the optimization pass of a compiler. Some involve recursively parsing regions. Tack on remainder, if any.
    sgs = {}
    sns = {}
    sgidents = []
    for sg in partition.strokeGroups:
        sgs[sg.ident] = sg
        sgidents.append(sg.ident)
        sns[sg.ident] = SymbolNode(sg, sns, sgs)

    parse = Parse(sgs)
    if len (sgs) == 0:
        parse.head = None
    else:
        #parse.head = l2r[0]
        #sns[parse.head] = SymbolNode(sgs[l2r[0]], sns, sgs) #pretend these are pointers to get how it works.

        recursiveParseReal(sgidents, sns, sgs)
        parse.head = getHead(sns, sgs)


    parse.sns = sns
    return parse


def recursiveParseReal(sgidents, sns, sgs): #the actual recursive part that tries to parse a set of sgidents.
    sgis = sgidents

    fracs = hLines(sgidents, sgs)
    while (len (fracs) > 0):
        tmp = fracs.pop(0)
        abv = strictlyAbove(tmp, sgis, sgs)
        blw = strictlyBelow(tmp, sgis, sgs)
        if(len(abv) > 0 and len(blw) > 0):
            for i in abv:
                sgis.remove(i)
                try:
                    fracs.remove(i)
                except ValueError:
                    pass

            for i in blw:
                sgis.remove(i)
                try:
                    fracs.remove(i)
                except ValueError:
                    pass

            recursiveParseReal(abv, sns, sgs)
            recursiveParseReal(blw, sns, sgs)

            assert(sns[tmp].above == None)
            assert(sns[tmp].below == None)
            sns[tmp].above = getHead(sns, sgs, abv)
            sns[tmp].below = getHead(sns, sgs, blw)

        
    #Handle roots pretty much the same way.        
    rts = roots(sgidents, sgs)
    while (len (rts) > 0):
        tmp = rts.pop(0)
        (inside, above) = inRoot(tmp, sgis, sgs)
        if(len(inside) > 0 or len(above) > 0):
            for i in inside:
                sgis.remove(i)
                try:
                    rts.remove(i)
                except ValueError:
                    pass

            for i in above:
                sgis.remove(i)
                try:
                    rts.remove(i)
                except ValueError:
                    pass

            recursiveParseReal(inside, sns, sgs)
            recursiveParseReal(above, sns, sgs) #completely overkill for the data we are working with.

            assert(sns[tmp].above == None)
            assert(sns[tmp].inside == None)
            if (len(above) > 0):
                sns[tmp].above = getHead(sns, sgs, above)
            if (len(inside) > 0):
                sns[tmp].inside = getHead(sns, sgs, inside)



    #Left to right is a reasonable default after all that stuff, I suppose.

    l2r = leftToRight(sgis, sgs)
    if len (sgis) != 0:        
        prev = l2r[0]
        for sgi in l2r[1:]:
            assert(sns[prev].right == None)
            sns[prev].right = sgi
            sns[sgi] = SymbolNode(sgs[sgi], sns, sgs)
            prev = sgi

    
        
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

    
def getHead(sns, sgs, idents = None):
    if idents == None:
        idents = list(sns.keys())
    parentless = set(idents)
    parented = set([])

    def procChild(id):
        if not id is None:
            if parented.intersection({id}) == {id}:
                None
                #print ("ERROR: not a tree!")
            elif parentless.intersection({id}) == {id}:
                parentless.remove(id)
                parented.add(id)
            #else:
                #print(id , " not anticipated.")

            
    for si in idents:
        sn = sns[si]
        procChild(sn.right)
        procChild(sn.above)
        procChild(sn.below)
        procChild(sn.inside)
        procChild(sn.subscript)
        procChild(sn.superscript)

    #if len(parentless) != 1:
        #print ("Warning: multiple trees.")
        #print ("roots: " , parentless)
        
    return parentless.pop()
            
            
SymbolClassesDict = {'baseline': ['\\alpha', '\\cos', '\\gamma', '\\infty', '\\pi', '\\sigma', '\\times', 'a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z'],
                     'ascender': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '\\pm', '\\forall', '\\in', '\\exists', '\\Delta', '\\theta', '\\lambda', 'i', '\\lim', '\\sin', '\\tan'],
                     'descender': ['\\beta', '\\mu', 'g', 'p', 'q', 'y', '\\mu'],
                     'extender': ['(', ')', '[', ']', '\\phi', 'j', '\\int', '\\log', '\\sum', '\\{', '\\}', '|'],
                     'centered': ['\\times', '\\div', '\\rightarrow', '-', '+'],
                     'root': ['\\sqrt'],
                     'punctuation': ['\\leq', '\\geq', '\\neq', '\\prime', '!', '/', '\\gt', '\\lt'],
                     'low': ['\\cdot', '\\ldots', 'COMMA'] }


def invertDict(d): #used to turn SymbolClassesDict into a form we can use to efficiently look up the category for a given class.
    items = d.items()
    n = {}
    for key, val in items:
        for thing in val:
            n[thing] = key
    return n

classCatDict = invertDict(SymbolClassesDict)


def calcCenterBox(sg, xmeandist, ymeandist):
    sg_xmin = sg.xmin()
    sg_xmax = sg.xmax()
    sg_ymin = sg.ymin()
    sg_ymax = sg.ymax()

    sg_xdist = sg.xdist()
    sg_ydist = sg.ydist()
    
    cat = classCatDict(sg.correctClass)

    
    
