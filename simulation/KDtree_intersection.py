# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:48:20 2020



Test pour savoir si les 2 segments ab et xy intersectent

D'après les NumericalRécipes 3rd Edition
section 21.4.1 Line Segment Intersections and “Left-Of” Relations



    

    

@author: ncharvin
"""


import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree



def BuildIntersectionArray(test_array):
    """
    Input 'test_array':
        2-D array constitué comme suit:
            u,v,ax,ay,bx,by,cx,cy,dx,dy     (10 colonnes)

            u : indice du premier segment 'ab' (indice qui se rapporte au tableau 'nanos')
            v : indice du premier segment 'cd' (indice qui se rapporte au tableau 'nanos')
            ax,ay : coordonnées du 1er  point du 1er segment
            bx,by : coordonnées du 2eme point du 1er segment
            cx,cy : coordonnées du 1er  point du 2éme segment
            dx,dy : coordonnées du 2eme point du 2ème segment
            
            
    Output:
        2-D array,  'subset' du tableau d'entrée 'test_array', 
        avec uniquement les paires de segments qui intersectent.
        Méthode basé sur les NumericalRecipes 3rd edition
        section 21.4.1 Line Segment Intersections and “Left-Of” Relations
    """

    ax = test_array[:,2]
    ay = test_array[:,3]
    bx = test_array[:,4]
    by = test_array[:,5]
    cx = test_array[:,6]
    cy = test_array[:,7]
    dx = test_array[:,8]
    dy = test_array[:,9] 
    

    denom = (dx - cx)*(ay-by) - (dy-cy)*(ax-bx)
    
    
    s = (ax-bx)*(cy-ay) - (ay-by)*(cx-ax)  
    t = (cx-ax) * (dy-cy)  -  (cy-ay)*(dx-cx)
    
    s = s / denom
    t = t / denom


    res = np.zeros_like(s)
    res[(s>=0) & (s <= 1) & (t>=0) & (t <= 1)]= 1
    

    out = test_array[res==1]


    
    
    """
    Rajoutons 2 colonnes à out, pour y écrire les coordonnées des points d'intersections
    
    		x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    		y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    """
    
    xi = np.zeros_like(out[:,0])
    yi = np.zeros_like(out[:,0])
    
    xi = ((out[:,2]*out[:,5] - out[:,3]*out[:,4]) * (out[:,6] - out[:,8]) - (out[:,2] - out[:,4]) * (out[:,6]*out[:,9] - out[:,7]*out[:,8])) / ((out[:,2] - out[:,4]) * (out[:,7] - out[:,9]) - (out[:,3] - out[:,5]) * (out[:,6] - out[:,8]));
    yi = ((out[:,2]*out[:,5] - out[:,3]*out[:,4]) * (out[:,7] - out[:,9]) - (out[:,3] - out[:,5]) * (out[:,6]*out[:,9] - out[:,7]*out[:,8])) / ((out[:,2] - out[:,4]) * (out[:,7] - out[:,9]) - (out[:,3] - out[:,5]) * (out[:,6] - out[:,8]));

    out = np.column_stack((out,xi,yi))

    return out




def ComputeIntersectionsFile_KD(segmentsFile='segments.txt', nw_length=8.0, intersectionsFile='intersections.txt'):
    
    
    #on efface l'ancien fichier intersections.txt
    try:    
        os.remove(intersectionsFile)
    except:
        pass
    
    
    #nanos = np.loadtxt(datafile, skiprows=1, delimiter='\t', usecols=(1,2,3,4))
    nanos_pd = pd.read_csv(segmentsFile, delimiter='\t', header=None, names=['x0','y0','x1','y1'], skiprows=1)
    nanos = nanos_pd.values
    # --> nanos = [x0,y0,x1,y1]
    
    Cx = (nanos[:,0]+nanos[:,2]) / 2
    Cy = (nanos[:,1]+nanos[:,3]) / 2
    
    
    nanos = np.column_stack((nanos,Cx,Cy))
    # --> nanos = [x0,y0,x1,y1,xc,yc]
    
    ckd=cKDTree(nanos[:,4:6])
    """
    'pairs_of_possibly_intersecting_segments' : tableau 2-D, qui contient 
    toutes les paires de segments (référencés par leurs indices dans la tableau 'nanos')
    qui possiblement se croisent.
    Ce tableau est crée grâce au KD-tree et à la longueur des nanofils.
    """
    
    pairs_of_possibly_intersecting_segments = ckd.query_pairs(r=nw_length,output_type='ndarray')
    
    
    # build possibly intersecting nanos array
    
    """
    J'initialise le tableau 'testing_array', qui sera constitué ainsi:
    testing_array = u,v,ax,ay,bx,by,cx,cy,dx,dy     (10 colonnes)
    
    u : indice du premier segment 'ab' (indice qui se rapporte au tableau 'nanos')
    v : indice du premier segment 'cd' (indice qui se rapporte au tableau 'nanos')
    ax,ay : coordonnées du 1er  point du 1er segment
    bx,by : coordonnées du 2eme point du 1er segment
    cx,cy : coordonnées du 1er  point du 2éme segment
    dx,dy : coordonnées du 2eme point du 2ème segment
        
    """
    testing_array=np.empty((len(pairs_of_possibly_intersecting_segments),10))
    # u,v, ax,ay,bx,by,cx,cy,dx,dy
    
    u = pairs_of_possibly_intersecting_segments[:,0]
    v = pairs_of_possibly_intersecting_segments[:,1]
    
    # ça a l'air stupide d'affecter u et v, mais je vais les utiliser plus bas
    # et ça sera plus lisible.
    testing_array[:,0] = u
    testing_array[:,1] = v

        # segment ab
    """
    testing_array[:,2] = nanos[u,0] 
    testing_array[:,3] = nanos[u,1]
    testing_array[:,4] = nanos[u,2]
    testing_array[:,5] = nanos[u,3]
    """
    testing_array[:,2:6] = nanos[u,0:4]
  
        
        # segment cd
    """
    testing_array[:,6] = nanos[v,0] 
    testing_array[:,7] = nanos[v,1] 
    testing_array[:,8] = nanos[v,2] 
    testing_array[:,9] = nanos[v,3]
    """
    testing_array[:,6:10] = nanos[v,0:4]

    
    # sur chaque ligne du fichier "segmentFiles", l'indice du nanofil et les coordonnées P1.x, P1.y, P2.x, P2.y
    # à la lecture, je squeeze l'indice du segment
    
    """
    seg_index,px0, py0, px1, py1 = np.loadtxt('segments.txt', skiprows=1, 
                                              delimiter='\t', unpack=True)
    """


    out  = BuildIntersectionArray(testing_array)

    out = out[np.lexsort((out[:,1],out[:,0]))]
    
    np.savetxt('intersections.txt', out[:,[0,1,10,11]], delimiter=" ", fmt='%d %d %.8e %.8e')
    # delimiter = ' ', to mimic Intersections_multithread_2019-01.exe 
    np.save('intersections.bin', out)

    nb_intersections = len(out)

    """
    Inserting nb of intersections as first line of intersections files
    Ugly! to be compatible with C-code method
    """
    with open(intersectionsFile, 'r+') as file:
        originalContent = file.read()
        file.seek(0, 0)              # Move the cursor to top line
        file.write('{0:d}\n'.format(nb_intersections))             # Add a new blank line
        file.write(originalContent)


    return  nb_intersections
    
    
