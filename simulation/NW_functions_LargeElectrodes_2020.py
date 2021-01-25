# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:41:39 2018

@author: Nicolas CHARVIN
LEPMI Laboratory:  https://lepmi.grenoble-inp.fr/
GUIDE team: http://www.lepmi-guide.univ-smb.fr
"""

import numpy as np
import scipy as sp
import scipy.interpolate as sci
import scipy.sparse as scsp

from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import collections as mplcollec
from mpl_toolkits.axes_grid1 import make_axes_locatable


import networkx as nx
import itertools

import sys
import os, time
import subprocess

from collections import OrderedDict
import pandas as pd

#import time, sys

import numba

import collections
#import shutil


def Generate_random_centers_and_angles(box_xsize=10, box_ysize=10, nw_length=1, n=10):
    """
    generate n (xc,yc,ac) tuples with random values:
        -electrodes_width <= xc  < box_xsize+electrodes_width
        0 <= yc  < box_ysize
        0 <= ac  < 3.14159
        
    Return:
        (xc values, 
    """
    xc_values = np.random.uniform(-nw_length, box_xsize+nw_length, size=n)
    yc_values = np.random.uniform(0, box_ysize, size=n)
    ac_values = np.random.uniform(0, np.pi, size=n)
    
    
    
    return xc_values, yc_values, ac_values


#   http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
# L'idée, pour 2 segments AB et CD, c'est de vérifier que le sens de rotation entre les segments (AB, AC) et
# (AB, AD) soient différents, et idem dans l'autre sens.

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def intersect(A, B, C, D):
    """
    Return True if segment AB intersects segment CD
    """
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def line_coefs(p1, p2):
    """    
    from Points p1 and p2
    return coeffs A,B,C of line equation: Ax + By = C
    """
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x * p2.y - p2.x * p1.y)
    return A, B, -C


def intersection_Coordinates(p1, p2, p3, p4):
    """
    # http://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
    return x,y  of intersection point from 4 Points
    """

    L1 = line_coefs(p1, p2)
    L2 = line_coefs(p3, p4)
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return None    # was return False






class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
       

    def isInsideRectangle(self, rect):
        """
        rect is a list[xleft, xright, ybottom, ytop]
        """
        xleft, xright, ybottom, ytop = rect 
        if (self.x >= xleft) and (self.x <= xright ) \
            and (self.y >= ybottom) and (self.y <= ytop ):
            return True
        else:
            return False


class Nanowire(object):

    def __init__(self, index=None, xc=0.0, yc=0.0, angle_rad=0.0, length=10.0):
        self.index = index
        self.xc = xc
        self.yc = yc
        self.angle_rad = angle_rad
        self.length = length        
        
        dummy = (length/2.0 * np.cos(angle_rad))
        x1 = xc - dummy
        x2 = xc + dummy
        
        dummy = (length/2.0 * np.sin(angle_rad))
        y1 = yc - dummy
        y2 = yc + dummy


        self.P1 = Point(x1, y1)
        self.P2 = Point(x2, y2)



def DumpNanowiresListToFile(nanowiresList, outfilename='segments.txt'):
    """
    Writing text file containing:
    1) the number of nanowires
    2) on each line:  NW index, coordinates P1.x, P1.y, P2.x, P2.y
    """
    
    print("Dumping Nanowires list to custom format...")

    fout = open(outfilename,'w')
    buf= "{0:d}\n".format(len(nanowiresList))    
    fout.write(buf)
    
    for nano in nanowiresList:
        buf = "{0:d}\t{1:.8e}\t{2:.8e}\t{3:.8e}\t{4:.8e}\n".format(nano.index, nano.P1.x, nano.P1.y, nano.P2.x, nano.P2.y)
        fout.write(buf)
    
    fout.flush()
    fout.close()
    return 0

def ReadNanowiresListFromFile(filename='segments.txt'):
    """
    Reading text file containing:
    1) the number of nanowires (which I skip)
    2) on each line:  NW index, coordinates P1.x, P1.y, P2.x, P2.y
    """

    print("Reading Nanowires list to custom format...")

    nano_indices,x1,y1,x2,y2 = np.loadtxt(filename, skiprows=1, unpack=True)
    nano_indices = nano_indices.astype('uint64')

    return (nano_indices,x1,y1,x2,y2)



def ComputeIntersectionsFile_C(segmentsFile='segments.txt', intersectionsFile='intersections.txt'):
    """
    From the nanowires list (read from file), we built an 'intersections' file, 
    using an external compiled C program (Microsoft Windows only)
    1) computing intersections
    2) writing resulting textfile, formatted as:
        number of intersections
        index_of_NW1  index_of_NW2  Xintersection   Yintersection
        ...
        ...
        
        
    Return:
        nb of intersections found
   """
    
    
    #removing former intersectionsFile
    try:    
        os.remove(intersectionsFile)
    except:
        pass
    
    
    cmd = "Intersections_multithread_2019-01.exe {0} 4".format(segmentsFile)    # 4 = nb threads
    status = subprocess.call(cmd,shell=True)
    if status != 0 :    
        print("CMD status", status)
        print("Error computing intersections. Isn't file {0} blocked ??".format(intersectionsFile))
        return(-12345)
    
    fin = open(intersectionsFile, 'r')
    nb_intersections = int(fin.readline())
    fin.close()
    
    return nb_intersections



def BuildNodes_and_Rcontacts_and_Rwires(intersectionsFile='intersections.txt'):
    """
    EN:
    From the textfile listing the intersections:
        Each line contains names of the 2 intersecting NW (say nanowire_A and
        nanowire_B), and intersection coordinates (xinter,yinter)
        From each intersection, we build TWO nodes (A_B) and (B_A), with the
        same (xinter,yinter), linked by a contact resistor Rc
        
    Nodes are stored in the 'nodes' OrderedDict:
        nodes[key] = (xinter, yinter)
            with key  = "nanowire1"_"nanowire2"
    
    Rc contact resistors are stored in 'rcontacts', a list of tuples.
    Each tuple contains the two keys of the two nodes linked together through
    this contact resistor.
    
    Rwire resistors are stored in a similar manner.
    
    
    FR:
    A partir de la liste des intersections:
        Chaque ligne nous donne le nanofilA et le nanofilB qui intersectent, 
        ainsi que les coords du point d'intersection.
        De chaque intersection, on tire DEUX noeuds (A_B) et (B_A), de même 
        coordonnées spatiales, et reliés par une résistance de contact Rc
        
    Les noeuds sont stockés dans le dictionnaire nodes:
        nodes[key] = (xinter, yinter)
        la clé = "indice nanofil 1"_"indice nanofil 2"
    
    Les résistance Rcontact sont stockées dans une liste de tuples appellée
    rcontacts : chaque tuple contient les 2 clés des 2 noeuds reliés par 
    une resistance de contact
        
    idem pour les résistances Rwires
    
    """
    L = []
    nodes = OrderedDict()
    rcontacts = []
    rwires = []

    f = open(intersectionsFile,'r')
    _ = int(f.readline())    # skipping the first line, containing nb of intersections
    
    
    """
    Building L, list of the intersections. Each element is a tuple composed
    of (NW1_name, NW2_name, xinter, yinter), from the intersectionsFile
    Casting to int() might be unnecessary 
    """
    while(1):
        line = f.readline()
        if line == "":
            break
        NW1,NW2,x,y = line.split()
        L.append((int(NW1),int(NW2),float(x),float(y)))
    f.close()



    """
    # lecture du fichier binaire, issu de Intersections_multithread_2019-01.exe
    import struct 
    L2 = []
    f=open('intersections.bin','rb')
    buf = f.read(struct.calcsize('<Q'))
    nb_intersections = struct.unpack('<Q', buf)[0]
    print(nb_intersections)
    for i in range(nb_intersections):
        buf = f.read(struct.calcsize('<QQff'))
        res = struct.unpack('<QQff', buf)
        #NW1,NW2,x,y = struct.unpack('<QQff', buf)
    L2.append((NW1,NW2,x,y))
    f.close()
    """



    """
    Creating nodes and Rcontact resistors
    """
    for (a,b,xinter,yinter) in L:
        # create Nodes
        # 2 noeuds differents, mais même coordonnées
        nodes[str(a)+'_'+str(b)] = {'x':xinter,'y':yinter}
        nodes[str(b)+'_'+str(a)] = {'x':xinter,'y':yinter}
        rcontacts.append((str(a)+'_'+str(b), str(b)+'_'+str(a)))
    
   
    """
    # 24.01.2019 :  il manque des Rwires
    # parce que c'est pas intersections.txt que je dois lire, mais la liste des Noeuds
    # issus de la création des Rcontact.
    # Intersections:  0, 3
                      0, 12
                      5, 6
                      12, 15
    et donc il y 2 fois plus de noeuds:
                      
     0_3
     3_0
     0_12
     12_0
     5_6
     6_5
     12_15
     
    Creating Rwire resistors
    This is done by scanning nodes names  (eg. 12_15, 15_12, 984_1237, 1237_984, ...),
    by spliting the name (sep = '_'), and sorting them (sort key = first part)
    
    Remember: node names 984_*** means that nodes are located on the same specific
    nanowire #984
    """
    
    nodes_list = list(nodes.keys())
    
    nodes_list_sorted = sorted(nodes_list, key=lambda x: x.split('_')[0])   
    #  VERY IMPORTANT TO SORT 
    #  OTHERWISE, itertools.groupby() does not work as expected
    
    for key,igroup, in itertools.groupby(nodes_list_sorted, key=lambda x: x.split('_')[0]):

        """
        igroup is a generator
        I must create Rwires from list(igroup)
        
        igroup est un iterateur.
        c'est à partir de ce list(igroup) que je dois créer mes Rwires !!!
        """
        list_nodes_on_nano_k = list(igroup)
        
        if len(list_nodes_on_nano_k) > 1:
            """
            Sorting nodes on this specific nanowire k, for them to be consecutive
            along the nanowires
            
            We sort the nodes according to their X position.
            What happens if the nanowire is truly vertical ? This sorting might
            gave wrong results.
            So we sort them a second time, according to their Y position.
            
            - if the nanowire is of any orientation, 2nd sort won't change the order
            - if the nanowire is vertical, 1st sort (on X) won't work, but 
              2nd sort (on Y )will sort the nodes consecutively
            - if the nanowire is horizontal, 1st sort (on X) will work, and 
              2nd sort (on Y) won't change the existing order. TESTED !
            
            
             je trie les noeud sur ce nanofil suivant x croissant
             ---   que se passe-t-il pour les nanofil verticaux:
             le tri suivant x va foirer.
             Dois-je faire des cas particuliers pour ces 2 nanofils ?
             Et qui si un nanofil est aléatoirement vertical ?
             
             Je peux tricher en triant d'abord suivant x, puis suivant y:
                 si le nanofil est quelconque, le 2ème tri ne changera rien
                 si le nanofil est vertical, le 2ème tri remettra les noeuds dans l'ordre
                 si le nanofil est horizontal, le 2ème tri (sur Y) ne change pas l'ordre 
                     du 1er tri (sur X). Verifié sur un exemple trivial.

            """
            
            list_nodes_on_nano_k.sort(key=lambda val: nodes[val]['x'])
            list_nodes_on_nano_k.sort(key=lambda val: nodes[val]['y'])
        
            for i in range(len(list_nodes_on_nano_k)-1):
                rwires.append( (list_nodes_on_nano_k[i], list_nodes_on_nano_k[i+1]) )
    
    return nodes, rcontacts, rwires


def BuildGraph(nodes=None, rcontacts=None, rwires=None, rlineic=1.234, rc=0.1, lsx=100.0):
    """
    rc = contact resistance (in Ohms)
    rlineic = lineic wire resistance (Ohms.µm-1)
    """
    
    g = nx.Graph()



    """
    Sorting nodes along increasing X position
    The goal is to then get a conductance matrix that is as diagonal as 
    possible when we build it.
    Not sure if this works, nor if it really helps solving
    
    Je trie les noeuds suivant les X croissants
    L'idée, c'est d'avoir les noeuds dans l'ordre de X croissants
    pour que la matrice A soit la plus diagonale possible quand
    on la construit. Je ne sais pas si ça peut aider à sa résolution...
    """
    
    nodes_sorted = OrderedDict(sorted(nodes.items(), key=lambda x:x[1]['x']))

    for key,data in nodes_sorted.items():
        g.add_node(key, x=data['x'], y=data['y'])

    
    for node1,node2 in rcontacts:
        g.add_edge(node1, node2, Rval=rc, Rtype='Rcontact')


    eps = 1.0e-4  #1.234e-16    
    
    for node1,node2 in rwires:
        distance = np.sqrt((nodes[node1]['x']-nodes[node2]['x'])**2 + (nodes[node1]['y']-nodes[node2]['y'])**2)
        

        """
        If two nodes are very close to each other, their distance might 
        numerically equals to zero. Which leads to a rwire resistance value Rval
        of 0 Ohms. Which in turn lead to real trouble in the conductance matrix
        building, because we use 1/Rval. 
        To avoid this issue, we always add a small epsilon to Rval.
        
        BUT !!
        
        If epsilon is too small, then 1/Rval became too large, and then
        the conductance matrix is ill-conditionned and might cannot be solved.
        
        Tests lead to a good epsilon value of 1e-4
        
        
        Si les deux noeuds sont très proches, il se peut que numériquement,
        la distance soit égale à 0. Donc, la résistance = 0.0 également.
        Comme dans la matrice, on utilise 1/Rval, on risque d'avoir un souci
        Donc, on rajoute un epsilon petit pour éviter ce problème.
        
        MAIS !!
        Si epsilon est trop petit, alors 1/Rval devient trop grand, et ça 
        empêche la résolution de la matric, qui est alors très très 
        "ill-conditionned" (déjà que c'est pas terrible: c'est pas comme
        si on avait un joli maillage avec des triangles équilatéraux)
        
        eps = 1e-15  -->   BAD
        eps = 1e-4   -->   GOOD
        """
        
        rwire_value = (distance * rlineic) + eps
    
        # Comme le graph n'est pas orienté, si on a déja rajouté (nodeA , nodeB),
        # peu importe si on rajoute (nodeB, nodeA), c'est la même edge
        
        # as the graph is undirected, it does not matter if we add edge(nodeB, nodeA), 
        # even if we already have added edge(nodeA,nodeA). There is only one edge at the end.
        g.add_edge(node1, node2, Rval=rwire_value, Rtype='Rwire')
    

    return g



def RemoveDanglingNodes(graph):
    """
    Given a graph, remove nodes (except on electrodes) 
    who have zero or one neighbor
    
    It does this repeatedly until no such node can be found
    
    Modify graph in-place
    
    
    Update 2019: since there are 2 virtual nodes for GND and PLUS, checking that
    dangling nodes is on_electrode or not should not be necessary anymore...
    EXCEPT for the virtual nodes !!!!   So, let's keep this part
        
    
    Return: total number of nodes removed
    """
    nb_nodes_removed = 0
    while 1:
        toremove = [node for (node, nb_neighbors) in graph.degree() if nb_neighbors < 2]
        #toremove = [n for n in toremove if (n.startswith('1_') == False) ]
        #toremove = [n for n in toremove if (n.startswith('0_') == False) ]
        
        toremove = [n for n in toremove if graph.nodes[n]['on_electrode'] == None ]
        
        if len(toremove) == 0:
            break
        graph.remove_nodes_from(toremove)
        nb_nodes_removed += len(toremove)
    
    return nb_nodes_removed





def RemoveOrphanNodes(graph):
    """
    Given a graph, remove nodes who have ZERO neighbor
    
    Modify graph in-place
    
    Return: total number of nodes removed
    """
    
    toremove = [node for (node, nb_neighbors) in graph.degree() if nb_neighbors < 1]
    graph.remove_nodes_from(toremove)
    
    return len(toremove)



def RemoveIsolatedLoops(graph):
    """
    Given a graph, percolating from GND to PLUS
    

    If my graph contains isolated loops ( that is all the connected components that are not 
    part of the percolating cluster ), then the corresponding KCL matrix became singular
    and cannot be solved.
    Theses isolated loops are not removed while cleaning dangling and orphan nodes, so we
    need to explicitely remove them.

    
    1) Find GND and PLUS nodes  indices
    2) Remove all the isolated loops
    
    Modify graph in-place     
                                      
    Return: None
    """
    
    GND_node_id = [n for n,d in graph.nodes(data=True) if d['on_electrode'] == 'virtual_GND'][0]
    PLUS_node_id = [n for n,d in graph.nodes(data=True) if d['on_electrode'] == 'virtual_PLUS'][0]
    
    for h in nx.connected_components(graph):
        if (GND_node_id in h) and (PLUS_node_id in h):
            break

    graph.remove_nodes_from( [n for n in list(graph.nodes()) if n not in h ] )
    
  
    
    return h


def RemoveIsolatedLoops_old(graph):
    """
    Given a graph, percolating from GND to PLUS
    
    Since the detection of the percolating cluster use the indices of virtual Nodes,
    nodes indices has to be relabeled continuously before.
    
    
    1) Relabel nodes, so that there is no gap, from 0 to "len(G.nodes())-1"
    2) Remove all the isolated loops (that is all the connected components that are not 
                                      part of the percolating cluster)
    
    3) Relabel nodes once again, since we may have removed nodes
    
    
    Return: graph modified, with isolated loops removed
    """
    
    graph = nx.convert_node_labels_to_integers(graph, label_attribute='node_string',
                                           ordering='sorted')
    
    for h in nx.connected_components(graph):
        if (0 in h) and (len(graph.nodes())-1  in h):
            break

    graph.remove_nodes_from( [n for n in list(graph.nodes()) if n not in h ] )
    

    graph = nx.convert_node_labels_to_integers(graph, label_attribute='node_string',
                                           ordering='sorted')    
    
    return graph




#@profile
def BuildMatrixandSolve(graph, Isource_level=1.0, NEW_METHOD=False):
    """
    Build conductance matrix A and currents vector b, and then solve Ax=b
    to obtain voltages vector.
    Nodes labels [0 -->  len(G.nodes())-1] give the indices of the matrix
    elements, and also the solution vector indices
    
    
    Construit la matrice A et le vecteur b, puis résouds Ax = b
    Les labels des noeuds [0 ...  len(G.nodes())-1 ] donnent les
    indices de la matrice A, ainsi que les indices du vecteur solution
    """

    dim = len(graph.nodes())
    A = scsp.eye(dim,dim, format='lil')
    b = scsp.lil_matrix((dim,1), dtype='float')


    if NEW_METHOD:
        """
        Bof, pas flagrant. C'est l'itération sur les arêtes qui prend du temps, et pas le remplissage de 
        la matrice (même si indexer la matrice par un tableau d'indices va plus vite que indice par indice ') 
        """
        indices = [i for i in graph.nodes()]
        
        values = [ sum( [ 1.0/d['Rval']  for u,v,d in graph.edges(nbunch=i,data=True)] ) for i in indices]
        A[indices, indices] = values
    
        for i in graph.nodes():
            for neigh in graph.neighbors(i):
                A[i, neigh] = -1.0/(graph[i][neigh]['Rval'])

    else:        
        for i in graph.nodes():
            value = sum( [ 1.0/d['Rval']  for u,v,d in graph.edges(nbunch=i,data=True)] )
            A[i, i] = value
    
            for neigh in graph.neighbors(i):
                A[i, neigh] = -1.0/(graph[i][neigh]['Rval'])
    
   
 

    
    
    
    b[-1] = Isource_level

    """
    Mais on a une contrainte supplémentaire, c'est que V(noeud 0) = 0
    
    Donc, on peut supprimer la 1er ligne et la 1er colonne de A, ainsi que la
    première valeur de b.
    
    """
    A = A[1:,1:]
    b = b[1:]


    # pas forcément nécessaire, mais ça évite un warning
    A = A.tocsr(copy=True)
    
    """
    on résoud ce systeme Ax=b, d'abord avec un solveur direct
    (on n'oublie pas de rajouter 0.0 au vecteur solution des tensions,
     correspondant au noeud GND)
    """

    solution = spsolve(A,b)
    solution = np.concatenate(([0.0],solution))
    return solution,A,b



def BuildMatrixandSolve_NEW(graph, Isource_level=1.0, NEW_METHOD=False):
    """
    NEW:
    on retourne la matrice de conductance complète (nbnodes*nbnodes, GND compris)
    idem pour le RHS
    
    
    Build conductance matrix A and currents vector b, and then solve Ax=b
    to obtain voltages vector.
    Nodes labels [0 -->  len(G.nodes())-1] give the indices of the matrix
    elements, and also the solution vector indices
    
    
    Construit la matrice A et le vecteur b, puis résouds Ax = b
    Les labels des noeuds [0 ...  len(G.nodes())-1 ] donnent les
    indices de la matrice A, ainsi que les indices du vecteur solution
    """

    dim = len(graph.nodes())
    A = scsp.eye(dim,dim, format='lil')
    b = scsp.lil_matrix((dim,1), dtype='float')


    if NEW_METHOD:
        """
        Bof, pas flagrant. C'est l'itération sur les arêtes qui prend du temps, et pas le remplissage de 
        la matrice (même si indexer la matrice par un tableau d'indices va plus vite que indice par indice ') 
        """
        indices = [i for i in graph.nodes()]
        
        values = [ sum( [ 1.0/d['Rval']  for u,v,d in graph.edges(nbunch=i,data=True)] ) for i in indices]
        A[indices, indices] = values
    
        for i in graph.nodes():
            for neigh in graph.neighbors(i):
                A[i, neigh] = -1.0/(graph[i][neigh]['Rval'])

    else:        
        for i in graph.nodes():
            value = sum( [ 1.0/d['Rval']  for u,v,d in graph.edges(nbunch=i,data=True)] )
            A[i, i] = value
    
            for neigh in graph.neighbors(i):
                A[i, neigh] = -1.0/(graph[i][neigh]['Rval'])
    
   
 

    
    
    
    b[-1] = Isource_level

    """
    Mais on a une contrainte supplémentaire, c'est que V(noeud 0) = 0
    
    Donc, on peut supprimer la 1er ligne et la 1er colonne de A, ainsi que la
    première valeur de b.
    
    """
    A_reduced = A[1:,1:]
    b_reduced = b[1:]


    # pas forcément nécessaire, mais ça évite un warning
    A_reduced = A_reduced.tocsr(copy=True)
    
    """
    on résoud ce systeme Ax=b, d'abord avec un solveur direct
    (on n'oublie pas de rajouter 0.0 au vecteur solution des tensions,
     correspondant au noeud GND)
    """

    solution = spsolve(A_reduced,b_reduced)
    solution = np.concatenate(([0.0],solution))
    return solution,A,b




def Interpolate_XYV(XYV_filename="XYV.txt", grid_size=500):
    """
    A partir d'un fichier de points "scattered" XYV,
    retourne un tuple (xi,yi,vi) de points vi interpoles sur une grille
    xi,yi de (grid_size * grid_size) elements
    """
    try:
        x,y,v = np.loadtxt(XYV_filename, unpack=True)
    except UnicodeDecodeError:
        data = np.load(XYV_filename)
        x = data['x']
        y = data['y']
        v = data['v']
    
    xi,yi = np.mgrid[x.min():x.max():grid_size*1.0j , y.min():y.max():grid_size*1.0j]
    vi = sci.griddata((x,y),v,(xi,yi), method='linear')

    
    
    return (xi,yi,vi)

def PlotXiYiVi_PotentialMap(xi,yi,vi, myfontsize=15):
    fig, ax = plt.subplots(facecolor='grey')
    ax.set_facecolor('grey')    
    ax.set_title("Interpolated Potential Map", fontsize=myfontsize)
    ax.set_xlabel(u'X (µm)', fontsize=myfontsize)
    ax.set_ylabel(u'Y (µm)', fontsize=myfontsize)
    plt.axis('equal')    

    print("Nb interpolated points: ", xi.shape) 
    

    # https://stackoverflow.com/a/11620982/2435546
    vi_without_NAN = vi[~np.isnan(vi)]

    levels = np.linspace(vi_without_NAN.min(), vi_without_NAN.max(), 101)  #101    
    
    
    CS=plt.contourf(xi,yi,vi,levels,cmap=plt.cm.hot_r)
    
    cbar = plt.colorbar()
    cbar.set_label('Voltage (V)', fontsize=myfontsize)
    cbar.ax.tick_params(labelsize=myfontsize)


   
    #https://stackoverflow.com/a/19419899/2435546    
    # pour récupérer les coordonnées des lignes de niveaux
    contour_levels = np.linspace(vi_without_NAN.min(), vi_without_NAN.max(), 11)
    #print(contour_levels)
    CS = plt.contour(xi,yi,vi,contour_levels,colors='black',linewidths=0.2, linestyles='solid')    
    #plt.clabel(CS, inline=True, fontsize=10)
    
    

    """
    Attention, si on trace la grille d'interpolation,on ne voit plus la colormap,
    car la grille est trop serrée. Il faut zoomer    
    """
    #plt.scatter(xi,yi,marker='x',c='w',s=3)     
    
    plt.show()
    return 0



def Plot_Nanowires_List(nw_list,color='black', overplot=False, bg_color='grey', 
                        fig_title='', myfontsize=15):
    if overplot:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(facecolor=bg_color)
        ax.set_aspect('equal')
        ax.set_facecolor(bg_color)

    if fig_title == '':
        fig_title = 'Nanowires within sample: {0:d}'.format(len(nw_list))
    ax.set_title(fig_title, fontsize=myfontsize)
    ax.set_xlabel(u'X (µm)', fontsize=myfontsize)
    ax.set_ylabel(u'Y (µm)', fontsize=myfontsize)
    
    ax.tick_params(axis='both', which='major', labelsize=myfontsize)

    
    
    segments = []
    
    for nw in nw_list:
        x1 = nw.P1.x
        y1 = nw.P1.y
        x2 = nw.P2.x
        y2 = nw.P2.y
        segments.append([(x1,y1),(x2,y2)])
    
    segment_collection = mplcollec.LineCollection(segments, colors=color, linewidths=1.0) #0.5)
    ax.add_collection(segment_collection)
    ax.autoscale()
    ax.margins(0.1)


def Overplot_IntersectionPoints():
    a,b,x,y = np.loadtxt('intersections.txt', skiprows=1, unpack=True, delimiter=' ')
    plt.plot(x,y,'wo', markersize=1)
    ax = plt.gca()
    ax.autoscale()
    ax.margins(0.1)

def Overplot_ElectrodesArea(box_xsize=10, box_ysize=10, electrodes_width=10, myfontsize=15):
    ax = plt.gca()
    electrode_GND = matplotlib.patches.Rectangle((-electrodes_width,0),electrodes_width,box_ysize, linewidth=1, edgecolor='b', facecolor='b')
    ax.add_patch(electrode_GND)
    electrode_PLUS = matplotlib.patches.Rectangle((box_xsize,0),electrodes_width,box_ysize, linewidth=1, edgecolor='r', facecolor='r')
    ax.add_patch(electrode_PLUS)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_title('GND (blue) and PLUS (red) electrodes. Their widths are not to scale', fontsize=myfontsize)
    

def PlotGraph(graph, fig_title='Nice Plot'):

    fig,ax = plt.subplots(facecolor='grey')
    ax.set_aspect('equal')
    ax.set_xlabel(u'X (µm)')   # il faut définir la string en unicode, d'ou le u''
    ax.set_ylabel(u'Y (µm)')
    ax.set_title(fig_title)

    posi = {}
    node_names = {}
    for key in graph.nodes():
        posi[key] = (graph.nodes[key]['x'],graph.nodes[key]['y'])
        node_names[key] = key

    #edges_colors = [d['color'] for u,v,d in graph.edges(data=True)]

    if len(graph.nodes()) < 200:
        nx.draw_networkx_nodes(graph, pos=posi)    
        nx.draw_networkx_edges(graph, pos=posi) # edge_color=edges_colors)    
        nx.draw_networkx_labels(graph, pos=posi, labels=node_names)

    else:
        nx.draw_networkx_nodes(graph, pos=posi, ax=ax, node_size = 15.5) #0.01)  # node_size=1.5    
        nx.draw_networkx_edges(graph, pos=posi, ax=ax, width=0.5)  # edge_color=edges_colors, width = 0.3)         

    # https://stackoverflow.com/a/58174702
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.show()
    return fig,ax

def SaveScatterVoltages(graph):

    nodes_sorted = sorted(list(graph.nodes()))    
    X = np.zeros(len(nodes_sorted))
    Y = np.zeros(len(nodes_sorted))
    V = np.zeros(len(nodes_sorted))
    
    for n in nodes_sorted:
        X[n]=graph.nodes[n]['x']
        Y[n]=graph.nodes[n]['y']
        V[n]=graph.nodes[n]['v']
    np.savez("XYV.npz",x=X,y=Y,v=V)



def SaveScatterCurrents(graph):

    X =[]
    Y =[]
    I =[]

    for u,v,data in graph.edges(data=True):
        X.append((graph.nodes[u]['x']+graph.nodes[v]['x']) / 2.0)
        Y.append((graph.nodes[u]['y']+graph.nodes[v]['y']) / 2.0)
        I.append( data['i'])
    
    np.savez("XYI.npz",x=X,y=Y,i=I)






def PlotScatterVoltage(xyv_file, figtitle='my title'):
    """
    Quid des potentiels aux bornes des résistances de contacts ?
    Lequel faut-il affiché ? Celui du dessus, ou celui du dessous, ou bien
    la moyenne des deux?
    """
    
    npzfile = np.load(xyv_file)
    X = npzfile['x']
    Y = npzfile['y']
    V_solution = npzfile['v']
    
    fig, ax = plt.subplots(facecolor='grey')    
        
    ax.set(title="Voltage Scatter "+figtitle, xlabel=u'X (µm)', ylabel=u'Y (µm)')
    ax.set_facecolor('grey')
    ax.set_aspect('equal')

    #plt.scatter(X,Y,c=V_solution,cmap='hot',s=0.5)    # or cmap='hot_r'
    plt.scatter(X,Y,c=V_solution,cmap='jet',s=0.5)    # or cmap='hot_r'
    
    cbar = plt.colorbar()
    cbar.set_label('Voltage (V)')
    
    
    """
    # pour modifier les labels de la colorbar
    
    #cbar_values=np.linspace(i_min, i_max, 5, endpoint=True)
    #cbar = plt.colorbar(ticks=cbar_values)
    cbar = plt.colorbar()
    cbar.set_label('Voltage (V)')
    #cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar_values])
    cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar.get_ticks()])
    """
    

def PlotScatterCurrent(xyi_file, figtitle='mytitle',normalize=True):
    """
    """
    npzfile = np.load(xyi_file)
    X = npzfile['x']
    Y = npzfile['y']
    I = npzfile['i']
    
    
    fig, ax = plt.subplots(facecolor='grey')    
    ax.set(title="Intensity Scatter "+figtitle, xlabel=u'X (µm)', ylabel=u'Y (µm)')
    ax.set_facecolor('grey')
    plt.axis('equal') 
    
   
    """
    Comme je trace des "gros" points, qui se chevauchent, je trie selon les
    I croissants, pour un meilleur rendu
    """
    d = np.column_stack((X,Y,I))
    d = d[np.argsort(d[:,2])]



    """
    on peut plotter en log sur la colorbar. Mais il faut alors spécifier une
    valeur minimale, puisque dans notre cas, I peut être très très petite (e-47),
    voire nulle, et donc, on verra rien.
    """
    ##plt.scatter(X,Y,c=I,cmap='hot',s=1.0, 
    ##            norm=matplotlib.colors.LogNorm(vmin=0.00001, vmax=max(I)))

    if normalize:
        plt.scatter(d[:,0],d[:,1], c=d[:,2],cmap='hot',s=1.0, 
                    norm=matplotlib.colors.Normalize(vmin=d[:,2].min(), vmax=d[:,2].max() ))

    #plt.scatter(d[:,0],d[:,1], c=d[:,2],cmap='hot',s=1.0, 
    #            norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=1e-1))
    
    plt.colorbar()
    plt.xlabel(u'X (µm)')   # il faut définir la string en unicode, d'ou le u''
    plt.ylabel(u'Y (µm)')

    
def PlotScatterResistances(graph):
    """
    
    """
    fig, ax = plt.subplots(facecolor='grey')    
    ax.set(title="Resistances Scatter", xlabel=u'X (µm)', ylabel=u'Y (µm)')
    ax.set_facecolor('grey')
    ax.set_aspect('equal')

    X = []
    Y = []    
    R = []
    
    for u,v,d in graph.edges(data=True):
        X.append((graph.nodes[u]['x']+graph.nodes[v]['x'])/2.0)
        Y.append((graph.nodes[u]['y']+graph.nodes[v]['y'])/2.0)
        R.append(d['Rval'])
    

    plt.scatter(X,Y,c=R,cmap='hot_r',s=0.5)
    plt.colorbar()


def PlotScatterVoltages_fromGraph(graph, myfontsize=25, show_cb=True):
    """
    Quid des potentiels aux bornes des résistances de contacts ?
    Lequel faut-il affiché ? Celui du dessus, ou celui du dessous, ou bien
    la moyenne des deux?
    """
    
    X = []
    Y = []
    V = []
    for n,d in graph.nodes(data=True):
        X.append(d['x'])
        Y.append(d['y'])
        V.append(d['v'])

    X = np.array(X)
    Y = np.array(Y)
    V = np.array(V)
   
    fig, ax = plt.subplots(facecolor='grey')    
        
    ax.set_xlabel(u'X (µm)', fontsize=myfontsize)
    ax.set_ylabel(u'Y (µm)', fontsize=myfontsize)
    
    ax.tick_params(axis='both', which='major', labelsize=myfontsize)
    
    ax.set_title("Voltage Scatter: step {0:05d}".format(graph.graph['step']), fontsize=myfontsize)
    ax.set_facecolor('grey')
    ax.set_aspect('equal')

    plt.scatter(X,Y,c=V,cmap='hot',s=3) #s=0.5   # or cmap='hot_r'
    
    if 'blue_dots' in graph.graph.keys():
        plt.plot(graph.graph['blue_dots'][:,0], graph.graph['blue_dots'][:,1],'bo')
    
    
    if show_cb:
        cbar = plt.colorbar()
        cbar.set_label('Voltage (V)', fontsize=myfontsize)
        cbar.ax.tick_params(labelsize=myfontsize)
    
    """
    # pour modifier les labels de la colorbar
    
    #cbar_values=np.linspace(i_min, i_max, 5, endpoint=True)
    #cbar = plt.colorbar(ticks=cbar_values)
    cbar = plt.colorbar()
    cbar.set_label('Voltage (V)')
    #cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar_values])
    cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar.get_ticks()])
    """
    
    #fig.set_tight_layout(True)
    #fig.set_size_inches((8.5,11), forward=True)
    #fig.set_size_inches((17,22), forward=True)

    
    #plt.tight_layout()
    
    """
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    """
    plt.show()
    




def PlotScatterVoltages_fromGraph_nonInteractive(graph, myfontsize=25, show_cb=True):
    """
    Quid des potentiels aux bornes des résistances de contacts ?
    Lequel faut-il affiché ? Celui du dessus, ou celui du dessous, ou bien
    la moyenne des deux?
    """
    
    X = []
    Y = []
    V = []
    for n,d in graph.nodes(data=True):
        X.append(d['x'])
        Y.append(d['y'])
        V.append(d['v'])

    X = np.array(X)
    Y = np.array(Y)
    V = np.array(V)
   
    fig, ax = plt.subplots(facecolor='grey')    
        
    ax.set_xlabel(u'X (µm)', fontsize=myfontsize)
    ax.set_ylabel(u'Y (µm)', fontsize=myfontsize)
    
    ax.tick_params(axis='both', which='major', labelsize=myfontsize)
    
    ax.set_title("Voltage Scatter: step {0:05d}".format(graph.graph['step']), fontsize=myfontsize)
    ax.set_facecolor('grey')
    ax.set_aspect('equal')

    plt.scatter(X,Y,c=V,cmap='hot',s=3)    # or cmap='hot_r'   s=0.5
    
    if 'blue_dots' in graph.graph.keys():
        plt.plot(graph.graph['blue_dots'][:,0], graph.graph['blue_dots'][:,1],'bo')
    
    
    if show_cb:
        
        #https://stackoverflow.com/a/26720422
        #cbar = plt.colorbar(fraction=0.046, pad=0.04)
        
        # https://stackoverflow.com/a/18195921
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.5 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        
        cbar = plt.colorbar(cax=cax)
        cbar.set_label('Voltage (V)', fontsize=myfontsize)
        cbar.ax.tick_params(labelsize=myfontsize)
    
    """
    # pour modifier les labels de la colorbar
    
    #cbar_values=np.linspace(i_min, i_max, 5, endpoint=True)
    #cbar = plt.colorbar(ticks=cbar_values)
    cbar = plt.colorbar()
    cbar.set_label('Voltage (V)')
    #cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar_values])
    cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar.get_ticks()])
    """
    
    
    matplotlib.use('agg')
    plt.ioff()
    
    return fig
    
    

def PlotRwiresCurrents_fromGraph(graph, figtitle='mytitle', inset=False, myfontsize=15):
    """
    A partir d'un graphe dont les noeuds possède un attribut 'v' (voltage)
    et des arêtes qui représentent les valeurs de résistance,
    on trace les résistances suivant l'intensité du courant qui les traverse
    """
    fig, ax = plt.subplots(facecolor='grey')    

    #ax.set_title("Rwires Current Map: "+figtitle, fontsize=myfontsize)
    ax.set_xlabel(u'X (µm)', fontsize=myfontsize)
    ax.set_ylabel(u'Y (µm)', fontsize=myfontsize)

    ax.tick_params(axis='both', which='major', direction='in', labelsize=myfontsize, width=myfontsize)

    ax.set_facecolor('grey')
    plt.axis('equal') 
    
    i_values = []
    for u,v,d in graph.edges(data=True):
        if (d['Rtype'] == 'Rwire'):
            i_values.append(d['i'])
    
    i_max = max(i_values)    
    i_min = min(i_values)
    #print ("PlotGraphCurrents: i_max= ", i_max, "i_min=", i_min)

    # https://stackoverflow.com/a/26562639/2435546
    # https://stackoverflow.com/questions/26545897
    
    norm = matplotlib.colors.Normalize(vmin=i_min,vmax=i_max)
    

    lines = []
    lines_colors = []
    i_values = []
    
    

    """
    for u,v,d in graph.edges(data=True):

        startx = graph.nodes[u]['x']
        starty = graph.nodes[u]['y']
        endx = graph.nodes[v]['x']
        endy = graph.nodes[v]['y']

        lines.append([(startx,starty), (endx,endy)])
            
        #plutôt que d'appeller to_rgba à chaque i_value, je fais une liste de i_values
        # et j'appelle to_rgba sur cette liste. Bcp plus rapide !
        #lines_colors.append(s_m.to_rgba(i_value))       
        i_values.append(d['i'])            
    """

    for u,v,d in graph.edges(data=True):
        if d['Rtype'] != 'Relectrode':
            startx = graph.nodes[u]['x']
            starty = graph.nodes[u]['y']
            endx = graph.nodes[v]['x']
            endy = graph.nodes[v]['y']
    
            lines.append([(startx,starty), (endx,endy)])
                
            #plutôt que d'appeller to_rgba à chaque i_value, je fais une liste de i_values
            # et j'appelle to_rgba sur cette liste. Bcp plus rapide !
            #lines_colors.append(s_m.to_rgba(i_value))       
            i_values.append(d['i'])            




    """
    Si on veut tracer les résistances par ordre de courant croissant (pour
    éviter des artefacts, où des resistances de courant faibles sont plottées
    "par dessus" des résistances de courants forts )
    
    https://stackoverflow.com/a/9764364
    """
    
    i_values,lines = (list(t) for t in zip(*sorted(zip(i_values, lines))))
    


    # choose a colormap
    #c_m = matplotlib.cm.CMRmap
    #c_m = matplotlib.cm.hot
    c_m = matplotlib.cm.jet

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    lines_colors = s_m.to_rgba(i_values)
    
    lc = mplcollec.LineCollection(lines, colors=lines_colors, linewidths=0.2) #0.5
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    """
    cbar = plt.colorbar(s_m)
    cbar.set_label('Current (A)')

    """   
    # pour modifier les labels de la colorbar
    """
    cbar_values=np.linspace(i_min*1000, i_max*1000, 5, endpoint=True)
    cbar = plt.colorbar(s_m, ticks=cbar_values)
    cbar.set_label('Current (mA)', fontsize=myfontsize)
    cbar.ax.set_yticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    
    cbar = plt.colorbar(s_m, cax=cax)
    cbar.set_label('Current (A)', fontsize=myfontsize)
    cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar.get_ticks()])
    cbar.ax.tick_params(labelsize=myfontsize)
    



        





    #####   WIP  #####
    axins = None

    
    if inset == True:
        """
        #ça, ça fonctionne
        
        fig2,ax2 = plt.subplots()
        xxx = np.linspace(0,10,100)
        ax2.plot(xxx, np.sin(xxx))
        
        print("INSET-------------------")
        axins = ax2.inset_axes([4,0,3,2], transform=ax2.transData)
        axins.plot(xxx, np.random.rand(xxx.shape[0]), 'r--')
        axins.set_xlim(0,1)
        axins.set_ylim(0,1)
        """
        
        print("INSET-------------------")
        
        
        
        
        # [xlower_left, ylower_left, xsize, ysize] , dans les coordonnées du graphe existant (grace à transform=ax.transData)
        # xlower_left et y_lowerleft correspondent au croisement des axes X et Y, et pas au coin inférieur gauche de la
        # boundingBox qui contient les axes, avec les ticks et les labels
        
        #axins = ax.inset_axes([160,250,150,120], transform=ax.transData)
        
        
        xlow,xhigh = ax.get_xlim()
        ylow,yhigh = ax.get_ylim()
        x0 = (xhigh-xlow)/2.0
        y0 = (yhigh-ylow)/2.0
        
        print("xlow,ylow,xhigh, yhigh", xlow, ylow, xhigh, yhigh)
        print("x0, y0", x0,y0)
        

        zorder = 11111
        axins=None
        
               
        inset_x_size = (xhigh-xlow)/2.0 * 0.6                 
        inset_y_size = (yhigh-ylow)/2.0 * 0.6
        axins = ax.inset_axes([x0,y0,inset_x_size,inset_y_size], transform=ax.transData)
        
        #axins.set_aspect('equal')   # pas sur que ça fonctionne...
        axins.set_facecolor('grey')
        

        # on est obligé de tout retracer, une limite de matplotlib 
        # avec la combinaison LineCollection et Inset
    
        c_m = matplotlib.cm.jet
        
    
        
        lc = mplcollec.LineCollection(lines, colors=lines_colors, linewidths=0.5)
        axins.add_collection(lc)
        
        axins.set_xlim(xlow*0.3,xhigh*0.3)
        axins.set_ylim(ylow*0.3,yhigh*0.3)

        


        """
        Pour mettre un fond blanc pour l'ensemble de l'inset (figure+axes+ticks), il
        semble qu'il n'y pas d'autres moyens que d'ajouter un patch Rectangle. Encore
        faut il le dessiner avec le bon Zorder 
        """
        
        
        #https://stackoverflow.com/a/41271773
        bb = axins.get_tightbbox(renderer=fig.canvas.get_renderer())
        bb_in_dataCoordinates = ax.transData.inverted().transform(bb)
        
        xbb0, ybb0 = bb_in_dataCoordinates[0,:]
        xbb1, ybb1 = bb_in_dataCoordinates[1,:]
                
        zorder = axins.get_zorder()
        zorder = zorder-1
        
        
        
        #myrect = plt.Rectangle(xy=(100,200), height=100, width=200, color='white', zorder=zorder)
        #myrect = plt.Rectangle(xy=(100,200), height=100, width=200, color='white', zorder=zorder)
        myrect = plt.Rectangle(xy=(xbb0,ybb0), width=(xbb1-xbb0), height=(ybb1-ybb0), color='white', zorder=zorder)
        
        ax.add_patch(myrect)

        
    #return fig,ax,axins
    return None


def PlotGraphRwiresPowers(graph, figtitle='mytitle', normalize=True):
    """
    A partir d'un graphe dont les noeuds possède un attribut 'v' (voltage)
    et des arêtes qui représentent les valeurs de résistance,
    on trace les résistances Rwires 
    suivant la puissance électrique qui les traverse
    """

    fig, ax = plt.subplots(facecolor='grey')    
    ax.set(title="Rwires Power Map "+figtitle, xlabel=u'X (µm)', ylabel=u'Y (µm)')
    ax.set_facecolor('grey')
    plt.axis('equal') 
    
    p_values = []
    for u,v,d in graph.edges(data=True):
        if (d['Rtype'] == 'Rwire'):
            p_values.append(d['p'])
    
    p_max = max(p_values)    
    p_min = min(p_values)  
    print ("PlotGraphPowers:  p_max= ", p_max, "p_min=", p_min)

    # https://stackoverflow.com/a/26562639/2435546
    # https://stackoverflow.com/questions/26545897
    
    norm = matplotlib.colors.Normalize(vmin=p_min,vmax=p_max)
    
    

    lines = []
    lines_colors = []
    p_values = []
    
    for u,v,d in graph.edges(data=True):
        if (d['Rtype'] == 'Rwire'):
            startx = graph.nodes[u]['x']
            starty = graph.nodes[u]['y']
            endx = graph.nodes[v]['x']
            endy = graph.nodes[v]['y']
    
            lines.append([(startx,starty), (endx,endy)])
                
            #plutôt que d'appeller to_rgba à chaque i_value, je fais une liste de i_values
            # et j'appelle to_rgba sur cette liste. Bcp plus rapide !
            #lines_colors.append(s_m.to_rgba(i_value))       
            p_values.append(d['p'])            


    if True: #normalize:
        p_values = np.array(p_values)
        p_values = (p_values - p_values.min()) /  (p_values.max() - p_values.min()) 
        print ("PlotGraphPowers Normalized (0-->1):  p_max= ", p_values.max(), "p_min=", p_values.min())
        norm = matplotlib.colors.Normalize(vmin=0.0,vmax=1.0)
        
        ax.set(title="Rwires Power Map Normalized(0-->1) "+figtitle, xlabel=u'X (µm)', ylabel=u'Y (µm)')

    
    # choose a colormap
    #c_m = matplotlib.cm.CMRmap
    c_m = matplotlib.cm.hot

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    lines_colors = s_m.to_rgba(p_values)
    
    lc = mplcollec.LineCollection(lines, colors=lines_colors, linewidths=0.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    #cbar = plt.colorbar(s_m)
    #cbar.set_label('Power (W)')

    # pour modifier les labels de la colorbar
    
    #cbar_values=np.linspace(i_min, i_max, 5, endpoint=True)
    #cbar = plt.colorbar(s_m, ticks=cbar_values)
    cbar = plt.colorbar(s_m)
    cbar.set_label('Power (W)')
    #cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar_values])
    cbar.ax.set_yticklabels(["{:.1e}".format(i) for i in cbar.get_ticks()])




    
    plt.xlabel(u'X (µm)')   # il faut définir la string en unicode, d'ou le u''
    plt.ylabel(u'Y (µm)')

    return (p_values)


def PlotGraphRwiresPowers_WhiteLargest(graph, figtitle='mytitle', normalize=True):
    """
    A partir d'un graphe dont les noeuds possède un attribut 'v' (voltage)
    et des arêtes qui représentent les valeurs de résistance,
    on trace les résistances Rwires 
    suivant la puissance électrique qui les traverse
    
    Toutes les resistances Rwires sont en noir, sauf celle dont la puissance
    est maximale, tracée en blanc
    
    """

    fig, ax = plt.subplots(facecolor='grey')    
    ax.set(title="Rwires Power Map "+figtitle, xlabel=u'X (µm)', ylabel=u'Y (µm)')
    ax.set_facecolor('grey')
    plt.axis('equal') 
    
    p_max = -9999
    for u,v,d in graph.edges(data=True):
        if (d['Rtype'] == 'Rwire'):
            if d['p'] > p_max:
                u_p_max = u
                v_p_max = v
                p_max = d['p']
    
    print ("PlotGraphPowers:  p_max= ", p_max)

    # https://stackoverflow.com/a/26562639/2435546
    # https://stackoverflow.com/questions/26545897
    
#    norm = matplotlib.colors.Normalize(vmin=p_min,vmax=p_max)
    
    

    lines = []
    
    for u,v,d in graph.edges(data=True):
        if (d['Rtype'] == 'Rwire'):
            startx = graph.nodes[u]['x']
            starty = graph.nodes[u]['y']
            endx = graph.nodes[v]['x']
            endy = graph.nodes[v]['y']
    
            lines.append([(startx,starty), (endx,endy)])
                
            #plutôt que d'appeller to_rgba à chaque i_value, je fais une liste de i_values
            # et j'appelle to_rgba sur cette liste. Bcp plus rapide !
            #lines_colors.append(s_m.to_rgba(i_value))       
            #p_values.append(d['p'])            

   
    # choose a colormap
    #c_m = matplotlib.cm.CMRmap
    #c_m = matplotlib.cm.hot

    # create a ScalarMappable and initialize a data structure
    #s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    #s_m.set_array([])

    #lines_colors = s_m.to_rgba(p_values)
    colors = ['black' for dummy in range(len(lines))]
    lc = mplcollec.LineCollection(lines, colors=colors, linewidths=0.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    # on trace en blanc la résistance Rwire avec la plus grande puissance
    startx = graph.nodes[u_p_max]['x']
    starty = graph.nodes[u_p_max]['y']
    endx = graph.nodes[v_p_max]['x']
    endy = graph.nodes[v_p_max]['y']
    
    plt.plot([startx,endx], [starty,endy], "w-")


    

    plt.xlabel(u'X (µm)')   # il faut définir la string en unicode, d'ou le u''
    plt.ylabel(u'Y (µm)')




def ExportScatterVoltages_fromGraph(graph):
    X = []
    Y = []
    V = []
    for n,d in graph.nodes(data=True):
        X.append(d['x'])
        Y.append(d['y'])
        V.append(d['v'])

    X = np.array(X)
    Y = np.array(Y)
    V = np.array(V)
 
    
 
    if 'blue_dots' in graph.graph.keys():
        blue_dots = graph.graph["blue_dots"]
        Xbd = blue_dots[:,0]
        Ybd = blue_dots[:,1]
    else:
        Xbd = np.array([np.nan])
        Ybd = np.array([np.nan])

    
    # https://stackoverflow.com/a/33404243
    # pas possible d'utiliser np.column_stack brutalement, car les colonnes n'ont pas les mêmes longueurs
    
    #d1 = {'X':X, 'Y':Y, 'V':V}
    #d2 = {'Xbd':Xbd, 'Ybd':Ybd}
    # J'utilise des OrderedDict, pour préserver l'ordre des colonnes dans le DataFrame.
    # Rappel:  les dict{} ne préservent pas l'ordre
    
    d1 = OrderedDict({'X':X, 'Y':Y, 'V':V})
    d2 = OrderedDict({'Xbd':Xbd, 'Ybd':Ybd})
    df1 = pd.DataFrame.from_dict(d1)
    df2 = pd.DataFrame(data=d2)
    df = pd.concat([df1, df2], axis=1)
    
    df.to_csv("Export_XYVblues_{0:05d}.txt".format(graph.graph['step']), sep='\t', index=False)
    
    return df
    







    

    
def ANSYS_Remove_Rcontacts_from_Graph(G):
    """
    supression des Rcontacts, et fusion des noeuds
    Pour ANSYS
    Entrée:  
        - le graph AVANT relabeling des noeuds ('aaa_bbb' --> integers )
        appellé 'G_avant_conversion' dans le script NW_simulation_LargeElectrodes.py'
        nécessaire pour supprimer les noeuds virtuels 
    
    
    """
    
    my_mapping = {}

    for u,v,d in G.edges(data = True):
        if d['Rtype'] == 'Rcontact':
            my_mapping[u]=v     # je vais fusionner u et v, les noeuds de part et d'autre d'une Rcontact
            
    H = nx.relabel_nodes(G,my_mapping)
    """
    je "traduis" les nodes labels, MINCE, ça ne fusionne pas:
    A  <---> B <----> C <---> D    : 
        4 noeuds : A,B,C,D
        3 edges (A-B),(B-C) et (C-D)
    
    Si je renomme C en B, le graphe ne contient plus que 3 noeuds A,B et D
    mais j'ai toujours 3 edges:
        A<--->B,  B<-->B  et B<-->D
        
    Je dois supprimer les edges à un seul noeud, d'où le to_remove
        
    """
    edges_to_remove = []
    for u,v in H.edges():
        if u == v:
            edges_to_remove.append((u,v))
    
    H.remove_edges_from(edges_to_remove)

    H.remove_node('GND')
    H.remove_node('PLUS')
    
    # je reconverti les nodes labels en entiers
    H = nx.convert_node_labels_to_integers(H, first_label=1)

    fout = open("ANSYS_edges.txt",'w')
    buf = ""
    for u,v in H.edges():
        buf += "{0}\t{1}\n".format(u,v)
    fout.write(buf)
    fout.close()

    fout = open("ANSYS_keypoints.txt",'w')
    buf = ""
    for n,d in H.nodes(data=True):
        buf += "{0:1.15e}\t{1:1.15e}\n".format(d['x'], d['y'])
    fout.write(buf)
    fout.close()        

    return H

    
def ANSYS_BuildNodes_and_Rwires(intersectionsFile='intersection.txt'):
    """
    POUR ANSYS: pas de résistances de contact
    
    
    A partir de la liste des intersections:
        Chaque ligne nous donne le nanofilA et le nanofilB qui intersectent, 
        ainsi que les coords du point d'intersection.
        De chaque intersection, on tire 1 noeud (A_B) et (B_A)
        
    Les noeuds sont stockés dans le dictionnaire nodes:
        nodes[key] = (xinter, yinter)
        la clé = "indice nanofil 1"_"indice nanofil 2"
    
    
    
    
    Les résistance Rcontact sont stockées dans une liste de tuples appellée
    rcontacts : chaque tuple contient les 2 clés des 2 noeuds reliés par 
    une resistance de contact
        
    idem pour les résistances 
    
    """
    L = []
    nodes = OrderedDict()
    rwires = []

    f = open('intersections.txt','r')
    nb_intersections = int(f.readline())    # to skip the first line
    
    """
    on crèe la liste L des intersections: chaque élement de cette liste est
    un tuple (indice_NanofilA, indice_NanofilB, x_intersection, y_intersection)
    
    """
    while(1):
        line = f.readline()
        if line == "":
            break
        NW1,NW2,x,y = line.split()
        L.append((int(NW1),int(NW2),float(x),float(y)))
    f.close()


    """
    on crèe les noeuds et les resistances de contact
    """
    for (a,b,xinter,yinter) in L:
        # create Nodes
        nodes[str(a)+'_'+str(b)] = {'x':xinter,'y':yinter}

   
    """
    # 24.01.2019 :  il manque des Rwires
    # parce que c'est pas intersections.txt que je dois lire, mais la liste des Noeuds
    # issus de la création des Rcontact.
    # Intersections:  0, 3
                      0, 12
                      5, 6
                      12, 15
    et donc il y 2 fois plus de noeuds:
                      
     0_3
     3_0
     0_12
     12_0
     5_6
     6_5
     12_15
    """
    
    nodes_list = list(nodes.keys())
    
    nanowires_list = []
    for node in nodes_list:
        a,b = node.split('_')
        nanowires_list.append(a)
        nanowires_list.append(b)
    
    nanowires_list = list(set(nanowires_list))
    nanowires_list = sorted(nanowires_list)
    
    
    rwires = []
    for q in nanowires_list:
        ll = []
        for n in nodes:
            a,b = n.split('_')
            if (q == a) or (q == b):
                ll.append(n)
        if len(ll) > 1:
            ll = sorted(ll, key=lambda val:nodes[val]['x'])
            for i in range(len(ll)-1):
                rwires.append((ll[i], ll[i+1]))
    
    
    # Export pour ANSYS
    fansys = open("edges_ANSYS.txt",'w')
    for a,b in rwires:
        x1 = nodes[a]['x']
        y1 = nodes[a]['y']
        x2 = nodes[b]['x']
        y2 = nodes[b]['y']
        buf=("{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:1.9e}\n").format(x1,y1,x2,y2)
        fansys.write(buf)
    fansys.close()
        
    #return nodes, rwires
    return nodes, nanowires_list, rwires



def Record_Rmacros(step, density, lsX, lsY, lw, rlineic_VALUE, rc_VALUE, relectrode_VALUE, rmacro, v_source_level):
    
    if os.path.isfile("Rmacros.txt") == False:
        fout = open("Rmacros.txt",mode='w')
        buf = "step\tTimestamp\tDensity\tLsX(um)\tLsY(um)\tLw(um)\tRlineic(ohm/um)\tRc(ohm)\tRelectrode(ohm)\tRmacro(ohm)\tVsource_Level(V)\n"
        fout.write(buf)
        fout.close()

    fout = open("Rmacros.txt",mode='a')
    buf = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9:.4f}\t{10:.4f}\n".format(step,time.strftime("%Y-%m-%d-%H-%M-%S"),
           density, lsX, lsY,lw, rlineic_VALUE, rc_VALUE, relectrode_VALUE, rmacro,v_source_level)
    
    fout.write(buf)
    fout.close()

def IsGraphPercolating(graph):
    """
    check if graph contains virtual_GND and virtual_PLUS nodes,
    and if there is a path between them.
    
    Return True if graph is percolating
    """
    GND_present = PLUS_present = False
    
    for n,d in graph.nodes(data=True):
        if d['on_electrode']=='virtual_GND':
            GND_node = n
            GND_present = True
        if d['on_electrode']=='virtual_PLUS':
            PLUS_node = n
            PLUS_present = True

    if GND_present and PLUS_present:
        if nx.has_path(graph, GND_node, PLUS_node):
            return True
    
    
    return False




#   ██████╗ ██████╗ ███████╗ ██████╗ ██╗     ███████╗████████╗███████╗
#  ██╔═══██╗██╔══██╗██╔════╝██╔═══██╗██║     ██╔════╝╚══██╔══╝██╔════╝
#  ██║   ██║██████╔╝███████╗██║   ██║██║     █████╗     ██║   █████╗  
#  ██║   ██║██╔══██╗╚════██║██║   ██║██║     ██╔══╝     ██║   ██╔══╝  
#  ╚██████╔╝██████╔╝███████║╚██████╔╝███████╗███████╗   ██║   ███████╗
#   ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚══════╝
#                                                                      

def ___OBSOLETE__(x):
    return None


def RemoveEdgesOnElectrodes(graph):
    """
    Remove edges that form wire resistance on electrode.
    that is:
        edge(u,v)  if graph.nodes[u]['on_electrode'] == 'GND' and
                      graph.nodes[v]['on_electrode'] == 'GND'
                      
                    idem for "PLUS"
    """
    to_remove = []
    for (u,v) in graph.edges():
        if (graph.nodes[u]['on_electrode'] == graph.nodes[v]['on_electrode'] != None):
            to_remove.append((u,v))
    
    print("DEBUG: ", to_remove)
    graph.remove_edges_from(to_remove)
            


def RemoveSingleNodesOnElectrodes(graph,Relectrode_value):
    """
    Given a graph, remove nodes from electrodes who have only neighbor
    also on the electrode:
    
        A ----- B ---- C
        |              |
        D              E
        
        becomes
        
        A--------------C
        |              |
        D              E
        
    
    Modify graph in-place
    
    Return: total number of nodes removed
    """

    single_nodes_removed = 0

    nodes_on_GND = [node for (node,data) in graph.nodes(data=True) if data['on_electrode'] == 'GND']
        
    for node in nodes_on_GND:
        neighbors = list(nx.neighbors(graph,node))
        if len(neighbors) == 2:
            neigh1,neigh2 = neighbors
            if (graph.nodes[neigh1]['on_electrode'] == 'GND' ) and (graph.nodes[neigh2]['on_electrode'] == 'GND' ):

                graph.add_edge(neigh1, neigh2)
                graph[neigh1][neigh2]['Rtype'] = 'Rwire'
                
                graph[neigh1][neigh2]['Rval'] = Relectrode_value
                
                graph.remove_node(node)
                single_nodes_removed += 1
    
    
    nodes_on_PLUS = [node for (node,data) in graph.nodes(data=True) if data['on_electrode'] == 'PLUS']
        
    for node in nodes_on_PLUS:
        neighbors = list(nx.neighbors(graph,node))
        if len(neighbors) == 2:
            neigh1,neigh2 = neighbors
            if (graph.nodes[neigh1]['on_electrode'] == 'PLUS' ) and (graph.nodes[neigh2]['on_electrode'] == 'PLUS' ):
                
                graph.add_edge(neigh1, neigh2)
                graph[neigh1][neigh2]['Rtype'] = 'Rwire'
                graph[neigh1][neigh2]['Rval'] = 0.001
                
                
                graph.remove_node(node)
                single_nodes_removed += 1

    
    return single_nodes_removed