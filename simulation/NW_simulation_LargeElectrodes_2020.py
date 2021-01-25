# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:32:31 2018

@author: Nicolas CHARVIN
LEPMI Laboratory:  https://lepmi.grenoble-inp.fr/


2019.07.31:
    - Modification in the graph creation procedure:
        1) Randomly casting wires, and intersections computations
        2) Graph creation, with Rwire et Rcontact resistors
        3) RELABELING of graph noded to integers, starting from 1
        4) Adding node 0, virtual node, (GND)
        5) Adding last node, virtual node (PLUS)
            that way,node '0' and node 'len(G.nodes())-1' are always connected
            to the voltage power supply, and they correspond to the first/last line of
            the conductance matrix, and they also correspond to the
            first and last position in the voltage solution vector.
    

"""


import NW_functions_LargeElectrodes_2020 as NF
import KDtree_intersection as KDinter

import numpy as np
#import scipy as sp
#import scipy.sparse as scsp
import shutil
import cProfile, pstats
import pickle
import zipfile

import matplotlib.pyplot as plt
import sys
import time
import os
import optparse
import operator
import random


import networkx as nx

if float(nx.__version__) < 2.4:
    print("Upgrade to networkX v2.4 or newer.  Node are accessed by G.nodes[], and not G.node[] anymore")
    sys.exit(-1)


if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
    sys.exit(-1)




parser = optparse.OptionParser()


parser.add_option("--density", action="store", dest="density", type="float",
                  help=u"nanowire density= N * Lw^2 /  (Lsx * Lsy). Default = 10)",
                  metavar="DENS", default=10)


parser.add_option("--LsX", action="store", dest="LsX", type="float",
                  help=u"X System Size (micro-meters). Default = 200 µm)",
                  metavar="sX", default=200)

parser.add_option("--LsY", action="store", dest="LsY", type="float",
                  help=u"Y System Size (micro-meters). Default = 100 µm)", 
                  metavar="sY", default=100)

parser.add_option("--Lw", action="store", dest="Lw", type="float",
                  help=u"Nanowires length (micro-meters). Default = 8 µm", 
                  metavar="sLw", default=8.0)

parser.add_option("--Dw", action="store", dest="Dw", type="float",
                  help=u"Nanowires diameter (nano-meters). Default = 70 nm", 
                  metavar="sDw", default=70.0)

parser.add_option("--Rl", action="store", dest="Rlineic", type="float",
                  help=u"NW lineic resistance value (ohms/µm). Default = 5.135 ohms/µm", 
                  metavar="sRl", default=5.135)

parser.add_option("--Rc", action="store", dest="Rcval", type="float",
                  help=u"Contact resistance value (ohms). Default = 100 ohms", 
                  metavar="sRc", default=100)

parser.add_option("--Re", action="store", dest="Relectrode", type="float",
                  help=u"Electrode <--> NW resistance value (ohms). Default = 0.001 ohms", 
                  metavar="sRe", default=0.001)

parser.add_option("--Vsource", action="store", dest="VsourceLevel", type="float",
                  help=u"Voltage source level (Volts). Default = 1 Volt", 
                  metavar="sV", default=1)

		  
parser.add_option("-p", action="store_true", dest="plotting",
                  help="Plot figures", default=False)


parser.add_option("--nbSteps", action="store", dest="nbSteps", type="int",
                  help="Nb of evolution steps (default=0)",
                  metavar="NB", default=0)

parser.add_option("--profile", action="store_true", dest="profiling",
                  help="Enable Profiling", default=False)


parser.add_option("--alter_mode", action="store", dest="alter_mode", type="string",
                  help="alteration mode: P_Rwires <default>, I_Rwires, j_Rwires, j_Rall, P_Rcontacts, ou Rcontacts_curing_and_Rwires_Pkill",
                  default='P_Rwires')

parser.add_option("--nbAlter", action="store", dest="nbAlter", type="int",
                  help="Nb of altered resistances per evolution step (default=1)",
                  metavar="NB", default=1)

parser.add_option("--jThresh", action="store", dest="jThresh", type="float",
                  help="Current density threshold (A.m-2) (default=1.62e10)",
                  metavar="NB", default=1.62e10)



parser.add_option("--existing_G", action="store", dest="existing_G", type="string",
                  help="existing graph network filename", default='')

parser.add_option("--no-save-graph", action="store_true", dest="no_save_graph",
                  help="Disable saving graph to disk", default=False)


(options, args) = parser.parse_args()


plt.close('all')

LsX = options.LsX
LsY = options.LsY
Lw = options.Lw
Dw = options.Dw
Sw = (np.pi*(Dw*1e-9)**2)/4     #  Sw in m-2


Density = options.density
Rlineic_VALUE = options.Rlineic
Rc_VALUE = options.Rcval
Relectrode_VALUE = options.Relectrode
V_SOURCE_LEVEL = options.VsourceLevel * 1.0
V_SOURCE_LEVEL_INCR = V_SOURCE_LEVEL / 10.0

PLOTTING = options.plotting

nbwires = int(Density * (LsX*LsY) / (Lw*Lw))
nbSteps = options.nbSteps
PROFILING = options.profiling
EXISTING_G = options.existing_G
NO_SAVE_GRAPH = options.no_save_graph
alter_mode = options.alter_mode
nbAlter = options.nbAlter
jThresh = options.jThresh

if alter_mode not in ['P_Rwires', 'I_Rwires', 'j_Rwires', 'j_Rall', 'P_Rcontacts', 'Rcontacts_curing_and_Rwires_Pkill']:
    print("{0} is an invalid alter mode. Quitting...".format(alter_mode))
    sys.exit(-1)


if PROFILING:
    pr = cProfile.Profile()
    pr.enable()


if EXISTING_G != '':
    print("Reading existing graph from disk")
    G = nx.read_gpickle(EXISTING_G)
    
    print("Reading existing NWlist from disk")
    NWlist = pickle.load(open("original_NWlist.p","rb"))
    percolating = True
    
else:
    print("Generating {0} nanowires".format(nbwires))
    XC,YC, AC = NF.Generate_random_centers_and_angles(box_xsize=LsX, box_ysize=LsY, n=nbwires, nw_length=Lw)
    NWlist = []
    
    for i,(xc,yc,ac) in enumerate(zip(XC,YC,AC)):
        N = NF.Nanowire(index=i, xc=xc, yc=yc, length=Lw, angle_rad=ac)
        NWlist.append(N)
    
    
    
    
    """
    update 2020.04 :      
    We either use:
    - brute-force multithread compiled C code (limited to Windows, maybe 
                                               with MSVC runtime dependencies issues)
    - fully-python KD-tree search 
    """
    
    NF.DumpNanowiresListToFile(NWlist)
    

    
    
    if os.path.exists('Intersections_multithread_2019-01.exe'):
        
        print("\nComputing intersections with external C code")
        startime = time.time()
        nb_intersections = NF.ComputeIntersectionsFile_C(segmentsFile='segments.txt')
        stoptime =time.time()
        print("Computation of Intersections (external C Code).  Duration (s): {0:0.2f}".format(stoptime-startime))    
        print("NB intersections: ", nb_intersections)
    
    else:
        print("\nComputing intersections with KD-tree method")
        startime = time.time()
        nb_intersections = KDinter.ComputeIntersectionsFile_KD(segmentsFile='segments.txt')
        stoptime =time.time()
        print("Computation of Intersections (KD-Tree).  Duration (s): {0:0.2f}".format(stoptime-startime))    
        print("NB intersections: ", nb_intersections)
    
    
    
    if PLOTTING:
        NF.Plot_Nanowires_List(NWlist, color='black')
        NF.Overplot_IntersectionPoints()
        NF.Overplot_ElectrodesArea(LsX, LsY, electrodes_width=2*Lw)
    
    
    
    startime = time.time()
    Nodes, Rcontacts, Rwires = NF.BuildNodes_and_Rcontacts_and_Rwires()
    stoptime =time.time()
    print("Build Nodes,Rcontacts and Rwires. Duration (s): {0:0.2f}".format(stoptime-startime))
    
    
    
    """
    Graph Generation
    """
    
    startime = time.time()
    
    
    G = NF.BuildGraph(nodes=Nodes, rcontacts=Rcontacts, rwires=Rwires, 
                      rlineic=Rlineic_VALUE, rc=Rc_VALUE, lsx=LsX)
    G.graph['LsX'] = LsX
    G.graph['LsY'] = LsY
    G.graph['Lw'] = Lw
    G.graph['Dw'] = Dw
    G.graph['Sw'] = Sw
    G.graph['Rlineic'] = Rlineic_VALUE
    G.graph['Relectrode'] = Relectrode_VALUE
    G.graph['Rc'] = Rc_VALUE
    G.graph['density'] = Density
    G.graph['step']=0
    
    stoptime =time.time()
    
    print("Building Graph duration (s): {0:0.2f} ".format(stoptime-startime))
    
    if PLOTTING:
        NF.PlotGraph(G, fig_title="Graph original")
    
    """
    Electrodes are supposed infinitely large. All nodes whose x<0 (resp x>LsX) 
    are considered on the GND electrode(-) (resp PLUS electrode (+) )
    """
    for n,d in G.nodes(data=True):
        if d['x'] < 0.0:
            G.nodes[n]['on_electrode']='GND'
    
        elif d['x'] > LsX:
            G.nodes[n]['on_electrode']='PLUS'
        
        else:
            G.nodes[n]['on_electrode']=None
            
    
    """
    ENG:
    Converting node names (for example "4155_8745") to integers, starting at 1
    
    Next, we will add a '0' node, corresponding to the GND connector of the power supply
    We will also add a 'last' node, corresponding to the PLUS connector. 
    
    Then, we link all the graph nodes located on electrodes to the corresponding terminal node.
    
    FRA:
    Je convertis les labels des noeuds ("4155_8745" par exemple), en entiers,
    en commencant à 1. Et ensuite, je rajouterai un noeud "0", correspondant
    au GND de la source de courant. Et aussi un noeud, tout à la fin, correspondant
    à la borne PLUS
    
    Et puis je raccorde à ces 2 nouveaux noeuds, les noeuds situés sur les électrodes
    droite et gauche
    
    Changement par rapport à FORRO: Rval = Relectrode_VALUE et pas RC_value/nb d'intersections
    sur l'électrode...
    
    """
    
    G = nx.convert_node_labels_to_integers(G,first_label=1, label_attribute='node_string')

        
   # adding node 0
    G.add_node(0, x= -2*Lw, y=LsY/2.0, on_electrode='virtual_GND')
    nodes_on_electrode_GND = [n for n,data in G.nodes(data=True) if data['on_electrode']=='GND']
    
    for n in nodes_on_electrode_GND:
        G.add_edge(0, n, Rval=Relectrode_VALUE, Rtype='Relectrode')
    


    # adding last node, which is called "len(G.nodes())"  (since G does not contain it yet!)        
    # j'ajoute le dernier noeud, qui s'appelle len(G.nodes())  (puisque G ne le contient pas encore)
    G.add_node(len(G.nodes()), x= LsX+2*Lw, y=LsY/2.0, on_electrode='virtual_PLUS')

    
    nodes_on_electrode_PLUS = [n for n,data in G.nodes(data=True) if data['on_electrode']=='PLUS']
    
    
    for n in nodes_on_electrode_PLUS:
        # trick: added to G, the last node is now called "len(G.nodes())-1"
        # sioux: ajouté à G, le dernier noeud s'appelle désormais len(G.nodes())-1
        G.add_edge(len(G.nodes())-1, n, Rval=Relectrode_VALUE, Rtype='Relectrode')
        
    
    print('Node GND = ',G.nodes[0])
    print('Node V+  = ',G.nodes[len(G.nodes())-1])
    
    
    percolating = False
    
    
    """    
    Must I clean graph before adding defect ???


    if NF.IsGraphPercolating(G):
        percolating = True
        print("---   Le graph percole ! ---")
    
        NF.RemoveIsolatedLoops(G)
        
        print("Orphan cleaning before adding defect: ",NF.RemoveOrphanNodes(G), " orphan nodes were removed") 
        print("Dangling cleaning before adding defect: ",NF.RemoveDanglingNodes(G), " dangling nodes were removed") 

        
        
        G = nx.convert_node_labels_to_integers(G, label_attribute='node_string',
                                           ordering='sorted') 


    else:
        print("Graph not percolating !!!!  QUIT")
        sys.exit(-1)
    """
    

    """


#   █████╗ ██████╗ ██████╗     ██████╗ ███████╗███████╗███████╗ ██████╗████████╗
#  ██╔══██╗██╔══██╗██╔══██╗    ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝╚══██╔══╝
#  ███████║██║  ██║██║  ██║    ██║  ██║█████╗  █████╗  █████╗  ██║        ██║   
#  ██╔══██║██║  ██║██║  ██║    ██║  ██║██╔══╝  ██╔══╝  ██╔══╝  ██║        ██║   
#  ██║  ██║██████╔╝██████╔╝    ██████╔╝███████╗██║     ███████╗╚██████╗   ██║   
#  ╚═╝  ╚═╝╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   
#                                                                               
                                                                              
    Adding defect on the sample
    """    

    
    Defect_Type = ''
    G.graph['defect']=Defect_Type
    
    print("\nAdding defect: ", Defect_Type)
    
    if Defect_Type =='hole':
        # Defaut circulaire sans noeuds
        nodes_to_remove=[]
        for u,v,d in G.edges(data=True):
            hole_radius = 20
            hole_xpos=40
            hole_ypos=30
            
            if ((G.nodes[u]['x']-hole_xpos)**2 + (G.nodes[u]['y']-hole_ypos)**2) < hole_radius**2:
                nodes_to_remove.append(u)
        G.remove_nodes_from(nodes_to_remove)
        
        
    elif Defect_Type == 'slit':
        # Fente verticale
        nodes_to_remove=[]
        slit_xpos=LsX/3
        slit_width=10
        slit_xleft = slit_xpos - (slit_width/2.0)
        slit_xright = slit_xpos + (slit_width/2.0)
        slit_height= 20  # 0.80*LsY
    
        for u,v,d in G.edges(data=True):
            
            if (slit_xleft < G.nodes[u]['x'] < slit_xright) and ( G.nodes[u]['y'] < slit_height):
                nodes_to_remove.append(u)
        G.remove_nodes_from(nodes_to_remove)
        
    
    else:
        print("No Defect")



    if NF.IsGraphPercolating(G) == False:
        print("Not percolating anymore after adding defect on the sample !!!!  QUIT")
        sys.exit(-1)

    
        
    """
    Cleaning graph
    """
    
    print("Orphan cleaning after adding defect: ",NF.RemoveOrphanNodes(G), " orphan nodes were removed") 
    print("Dangling cleaning: ",NF.RemoveDanglingNodes(G), " dangling nodes were removed") 




    """
    
    #  ██████╗ ███████╗███╗   ███╗ ██████╗ ██╗   ██╗███████╗    ██╗███████╗██╗      █████╗ ███╗   ██╗██████╗ ███████╗
    #  ██╔══██╗██╔════╝████╗ ████║██╔═══██╗██║   ██║██╔════╝    ██║██╔════╝██║     ██╔══██╗████╗  ██║██╔══██╗██╔════╝
    #  ██████╔╝█████╗  ██╔████╔██║██║   ██║██║   ██║█████╗      ██║███████╗██║     ███████║██╔██╗ ██║██║  ██║███████╗
    #  ██╔══██╗██╔══╝  ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██╔══╝      ██║╚════██║██║     ██╔══██║██║╚██╗██║██║  ██║╚════██║
    #  ██║  ██║███████╗██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ███████╗    ██║███████║███████╗██║  ██║██║ ╚████║██████╔╝███████║
    #  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝    ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝
    #                                                                                                                
    
        ENG: NETTOYAGE
        When I remove one or more resistors (eg when adding an defect),
        there can be "loops of resistors" that are not connected to the percolating
        network anymore.
        These loops are not orphan nor dangling nodes, but the must be removed,
        otherwise, the conductance matrix to be solved is a non solvable singular
        matrix.


        FR:NETTOYAGE
        Attention, quand je supprime plusieurs résistance ( par exemple avec
        l'ajout d'un defaut), je peux me retrouver avec des "boucles" 
        de résistances qui ne font plus partie du reseau percolant, 
        mais qui ne sont pas des orphans nodes ou des dangling nodes pour autant
        Je dois les supprimer, sinon je me retrouve avec une matrix singulière
        non solvable
    
        Et une fois que j'ai supprimé des boucles (et donc des nodes), 
        il faut une nouvelle fois que je relabelise les nodes.
    """

    print("Isolated loop cleaning: ")    
    #G = NF.RemoveIsolatedLoops_old(G)
    NF.RemoveIsolatedLoops(G)

    G = nx.convert_node_labels_to_integers(G, label_attribute='node_string',
                                           ordering='sorted')

    print("Nb nodes: ", len(list(G.nodes()) ) )
    print("Nb edges: ", len(list(G.edges()) ) )


    #if not NO_SAVE_GRAPH:
    # Je veux toujours sauvegarder le graphe orginal, même si je
    # n'enregistre pas les autres graphes, pendant l'évolution
        
    print("\nWriting original graph to disk")
    nx.write_gpickle(G, "graph_original.gpickle")
    
    print("Writing original NWlist to disk\n")
    pickle.dump(NWlist, open("original_NWlist.p","wb"))




"""
Building KCL matrix, and solve it

"""



startime = time.time()

I_SOURCE_LEVEL = 1.0

vecV, matA, vecB = NF.BuildMatrixandSolve(G, Isource_level=I_SOURCE_LEVEL)

Vplus = vecV[-1]
Rmacro = Vplus / I_SOURCE_LEVEL   #  U = RI


"""
Scaling to Vsupply = V_SOURCE_LEVEL
"""
scaling_factor = V_SOURCE_LEVEL / Vplus
vecV = vecV * scaling_factor
I_macro = I_SOURCE_LEVEL * scaling_factor




"""
Updating Graph Attributes
=========================
                                                                
After solving the KCL matrix, let's update graph nodes with a new
attribute 'v', equals to the voltage on each node.

Let's also update edges, by adding two attributes: 'i' (current)
and 'p' (power)


Finally, let's add a new graph attribute 'V_source_level'

Of course, this has to be done each time we solve the KCL matrix !
"""

for index,v in enumerate(vecV):
    G.nodes[index]['v']=vecV[index]


for u,v,d in G.edges(data=True):
    G[u][v]['i'] = np.abs(G.nodes[u]['v']-G.nodes[v]['v']) / d['Rval']
    G[u][v]['j'] = G[u][v]['i'] / Sw
    G[u][v]['p'] = d['Rval'] * (G[u][v]['i'])**2

G.graph['V_source_level']=V_SOURCE_LEVEL
G.graph['Rmacro']=Rmacro
G.graph['step']= 0    



stoptime =time.time()
print("Building and Solving Matrix duration (s): {0:0.2f} ".format(stoptime-startime))
print("Rmacro :{0:.4f}".format(Rmacro))


###   DEbugging NAN, avec le defaut
if np.isnan(Rmacro):
    rand_value = random.randint(10,1500)
    shutil.copyfile("graph_original.gpickle", "graph_original_{0:04}.gpickle".format(rand_value) )
    shutil.copyfile("original_NWlist.p", "original_NWlist_{0:04}.p".format(rand_value) )



#NF.Record_Rmacros(0,Density, LsX, LsY,Lw, Rlineic_VALUE, Rc_VALUE, Relectrode_VALUE, Rmacro, V_SOURCE_LEVEL)
NF.Record_Rmacros(0,G.graph['density'], G.graph['LsX'], G.graph['LsY'],G.graph['Lw'], 
                  G.graph['Rlineic'], G.graph['Rc'], G.graph['Relectrode'], G.graph['Rmacro'], 
                  G.graph['V_source_level'])


#  Save X,Y,Voltages to a .npz file
NF.SaveScatterVoltages(G)
NF.SaveScatterCurrents(G)



if PLOTTING:
    NF.PlotScatterVoltage("XYV.npz")
    NF.PlotScatterCurrent("XYI.npz")
    
    #NF.PlotGraphCurrents(G)
    #NF.PlotGraph(G, fig_title='Graph Final (without orphan nor dangling nodes)')


IMAX_Rwires = -999
PMAX_Rwires = -999
IMAX_Rcontacts = -999
PMAX_Rcontacts = -999

for u,v,d in G.edges(data=True):
    if d['Rtype']=='Rwire':
        i = d['i']
        p = d['Rval']*i*i
        if i > IMAX_Rwires:
            IMAX_Rwires = i
        if p > PMAX_Rwires:
            PMAX_Rwires = p

    if d['Rtype']=='Rcontact':
        i = np.abs(vecV[u]-vecV[v]) / d['Rval']
        p = d['Rval']*i*i
        if i > IMAX_Rcontacts:
            IMAX_Rcontacts = i
        if p > PMAX_Rcontacts:
            PMAX_Rcontacts = p




shutil.copyfile("XYV.npz", "XYV_{0:>04d}.npz".format(0)) 

    
    
"""
   ███████╗██╗   ██╗ ██████╗ ██╗     ██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
#  ██╔════╝██║   ██║██╔═══██╗██║     ██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
#  █████╗  ██║   ██║██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
#  ██╔══╝  ╚██╗ ██╔╝██║   ██║██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
#  ███████╗ ╚████╔╝ ╚██████╔╝███████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
#  ╚══════╝  ╚═══╝   ╚═════╝ ╚══════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
#                                                                            
    
"""

                                                                 
killed_resistors_positionsX = []
killed_resistors_positionsY = []


for step in range(1, nbSteps+1):
    print ("Evolution: modifying/killing resistors")
  
    # What is the modification scheme:  deleting Rwire (current or power threshold), or Rcontact
    # Est-ce que je casse les Rwires (suivant le courant I ou la Puissance P),
    # ou les Rcontacts ?
    print("\nstep {0} :  alter_mode = {1}, nbAlter = {2}, Vsource = {3:.5f}\n".format(step,alter_mode, nbAlter, V_SOURCE_LEVEL))

    
    G.graph['alter_mode']=alter_mode
    G.graph['nbAlter']=nbAlter
    G.graph['jThresh']=jThresh



    """
    ##  PLOT des courants et des puissances
    
    NF.PlotGraphRwiresCurrents(G, figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}".format(step, alter_mode, nbAlter))
    plt.savefig("Igraph_step_{0:04d}.png".format(step), dpi=200)
    plt.close()

    #NF.PlotGraphRwiresPowers(G, figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}. Vsrc={3:.1f} V".format(step, alter_mode, nbAlter, G.graph['V_source_level']))
    NF.PlotGraphRwiresPowers_WhiteLargest(G, figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}. Vsrc={3:.1f} V".format(step, alter_mode, nbAlter, G.graph['V_source_level']))
    
    plt.savefig("Pgraph_step_{0:04d}.png".format(step), dpi=200)
    plt.close()
    """
    
    

    # Among Rwires, listing individuals currents, power and current density
    # greater than IMAX_Rwires (resp PMAX_Rwires, jThresh)
    # and the corresponding (u,v) edges
    
    # Dans les Rwires, je liste les courants, puissance et densité de courant individuels
    # supérieurs à IMAX_Rwires (resp PMAX_Rwires, jThresh)
    # et les arêtes (u,v) correspondantes
    i_edges_wires=[]
    p_edges_wires=[]
    j_edges_wires=[]
    
    
    print("DEBUG  2020.05.27 :   alter_mode = {0}".format(alter_mode))
    jList = [d['j'] for u,v,d in G.edges(data=True) if d['Rtype'] in ['Rwire','Rcontact'] ]
    print("For Rwires AND Rcontact : max(j) = {0:.2e} ,  jThresh = {1:.2e} ".format(max(jList), jThresh))
    print("FIN DEBUG  2020.05.27")
    
    
    for u,v,d in G.edges(data=True):
        if d['Rtype']=='Rwire':
            ##  2020.05.25 c'est débile de recalculer i et p, ils sont déjà stocké dans le graphe.
            i = np.abs(vecV[u]-vecV[v]) / d['Rval']
            p = d['Rval'] * i * i
            j = d['j']
            if i>= IMAX_Rwires:
                i_edges_wires.append( (i,(u,v)) )
            if p>= PMAX_Rwires:
                p_edges_wires.append( (p,(u,v)) )
            if j>= jThresh:
                j_edges_wires.append( (j,(u,v)) )
                
        
    # sorting along decreasing indivial current (resp power, current density)
    # je trie suivant le courant individuel décroissant (resp puissance, densité de courant)
    i_edges_wires = sorted(i_edges_wires, 
                           key=operator.itemgetter(0),reverse=True)

    p_edges_wires = sorted(p_edges_wires, 
                           key=operator.itemgetter(0),reverse=True)
    
    j_edges_wires = sorted(j_edges_wires, 
                           key=operator.itemgetter(0),reverse=True)


    # same for Rcontacts, if individual currents >= IMAX_Rcontacts
    # pareil pour les Rcontacts, si le courant individuel  >= à IMAX_Rcontacts
    i_edges_contacts=[]
    p_edges_contacts=[]
    
    for u,v,d in G.edges(data=True):
        if d['Rtype']=='Rcontact':
            i = np.abs(vecV[u]-vecV[v]) / d['Rval']
            p = d['Rval'] * i * i
            if i>=IMAX_Rcontacts:
                i_edges_contacts.append( (i,(u,v)) )
            if p>=PMAX_Rcontacts:
                p_edges_contacts.append( (i,(u,v)) )
        
    # je trie suivant le courant individuel décroissant
    i_edges_contacts = sorted(i_edges_contacts, 
                           key=operator.itemgetter(0),reverse=True)

    
    
    
    
    # Pour le mode j_Rall, je pourrais peut-être réutiliser des listes déjà
    # créées, mais soyons bourrins
    j_edges_all=[]
    for u,v,d in G.edges(data=True):
        j = d['j']
        if j>= jThresh:
            j_edges_all.append( (j,(u,v)) )

    
    # je trie suivant la densité de courant individuelle décroissant
    j_edges_all = sorted(j_edges_all, 
                           key=operator.itemgetter(0),reverse=True)

    
    

    print("len(i_edges_rwires),len(p_edges_rwires), len(j_edges_rwires),len(j_edges_rall), len(i_edges_rcontacts), len(p_edges_rcontacts):",
          len(i_edges_wires),len(p_edges_wires),len(j_edges_wires), len(j_edges_all), len(i_edges_contacts), len(p_edges_contacts))

 
    if alter_mode == 'I_Rwires':
        if len(i_edges_wires) > 0:
            alter_resistances = True
            # je prends les "nbAlter" premiere arêtes de la liste i_edges_wires
            i_edges_wires = i_edges_wires[:nbAlter]
        
            for i,(u,v) in i_edges_wires:
                print ("Selected edge(s): ", u,v)
                u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                killed_resistors_positionsX.append((u_x+v_x)/2.0)
                killed_resistors_positionsY.append((u_y+v_y)/2.0)
                G.remove_edge(u,v)
        else:
            alter_resistances = False
            print("Aucune Rwires dont le courant dépasse IMAX_Rwires.")
            print("Fin de l'évolution à tension constante!")
            print("On augmente la tension")
            V_SOURCE_LEVEL += V_SOURCE_LEVEL_INCR


    elif alter_mode == 'P_Rwires':
        if len(p_edges_wires) > 0:
            alter_resistances = True
            # je prends les "nbAlter" premiere arêtes de la liste p_edges_wires
            p_edges_wires = p_edges_wires[:nbAlter]
        
            for _,(u,v) in p_edges_wires:
                print ("Selected edge(s): {0}-{1} .  Rval= {2}".format(u,v,G[u][v]['Rval']) )
                u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                killed_resistors_positionsX.append((u_x+v_x)/2.0)
                killed_resistors_positionsY.append((u_y+v_y)/2.0)
                G.remove_edge(u,v)
        
                # ou alors que j'augmente la résistance
                #print("\n\n DEBUG\n : je ne tue pas la résistance, mais multiplie sa valeur par 1e6")
                #G[u][v]['Rval'] *= 1e6;
        

        else:
            alter_resistances = False
            print("Aucune Rwires dont la puissance dépasse PMAX_Rwires.")
            print("Fin de l'évolution à tension constante!")
            print("On augmente la tension")
            V_SOURCE_LEVEL += V_SOURCE_LEVEL_INCR


    elif alter_mode == 'j_Rwires':
        if len(j_edges_wires) > 0:
            alter_resistances = True
            # je prends les "nbAlter" premiere arêtes de la liste j_edges_wires
            j_edges_wires = j_edges_wires[:nbAlter]
        
            for _,(u,v) in j_edges_wires:
                print ("Selected edge(s): ", u,v)
                u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                killed_resistors_positionsX.append((u_x+v_x)/2.0)
                killed_resistors_positionsY.append((u_y+v_y)/2.0)
                G.remove_edge(u,v)
        
                # ou alors que j'augmente la résistance
                #print("\n\n DEBUG\n : je ne tue pas la résistance, mais multiplie sa valeur par 1e6")
                #G[u][v]['Rval'] *= 1e6;
        
    elif alter_mode == 'j_Rall':
        if len(j_edges_all) > 0:
            alter_resistances = True
            # je prends les "nbAlter" premiere arêtes de la liste j_edges_wires
            j_edges_all = j_edges_all[:nbAlter]
        
            for _,(u,v) in j_edges_all:
                print ("Selected edge(s): {0}-{1}. j_val={4:.3e},  Rtype={2},  Rval= {3:.3f}".format(u,v,G[u][v]['Rtype'], G[u][v]['Rval'],G[u][v]['j'] ) )
                u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                killed_resistors_positionsX.append((u_x+v_x)/2.0)
                killed_resistors_positionsY.append((u_y+v_y)/2.0)
                
                """ #  DEBUG
                if G[u][v]['Rtype'] == 'Rcontact':
                    print("KABOOM")
                    sys.exit(-1)
                """
                
                
                G.remove_edge(u,v)
        
                # ou alors que j'augmente la résistance
                #print("\n\n DEBUG\n : je ne tue pas la résistance, mais multiplie sa valeur par 1e6")
                #G[u][v]['Rval'] *= 1e6;


        else:
            alter_resistances = False
            print("Aucune Rwires dont la densité de courant dépasse jThresh.")
            print("Fin de l'évolution à tension constante!")
            print("On augmente la tension")
            V_SOURCE_LEVEL += V_SOURCE_LEVEL_INCR




    elif alter_mode == 'P_Rcontacts':
        if len(p_edges_contacts) > 0:
            alter_resistances = True
            # je prends les "nbAlter" premiere arêtes de la liste p_edges_wires
            p_edges_contacts = p_edges_contacts[:nbAlter]
        
            for _,(u,v) in p_edges_contacts:
                print ("Selected edge(s): ", u,v)
                u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                killed_resistors_positionsX.append((u_x+v_x)/2.0)
                killed_resistors_positionsY.append((u_y+v_y)/2.0)
                G.remove_edge(u,v)
        
                # ou alors que j'augmente la résistance
                #print("\n\n DEBUG\n : je ne tue pas la résistance, mais multiplie sa valeur par 1e6")
                #G[u][v]['Rval'] *= 1e6;
        

        else:
            alter_resistances = False
            print("Aucune Rcontacts dont la puissance dépasse PMAX_Rcontacts.")
            print("Fin de l'évolution à tension constante!")
            print("On augmente la tension")
            V_SOURCE_LEVEL += V_SOURCE_LEVEL_INCR



    elif alter_mode == 'Rcontacts_curing_and_Rwires_Pkill':
        alter_resistances = False

        if len(i_edges_contacts) > 0:
            nb_Rcontact_cured = 0
            for i,(u,v) in i_edges_contacts:
                if G[u][v]['Rval'] > 5: # 5 , valeur plancher
                    #print ("Rcontact_cured: ",u,v)
                    G[u][v]['Rval'] *= 0.75
                    nb_Rcontact_cured +=1
                    alter_resistances = True


        if len(p_edges_wires) > 0:
            nb_Rwire_killed = 0
            p_edges_wires = p_edges_wires[:nbAlter]
        
            for p,(u,v) in p_edges_wires:
                if p>=1.5*PMAX_Rwires:
                    #print ("Rwire killed: ", u,v)
                    u_x,u_y = G.nodes[u]['x'],G.nodes[u]['y']
                    v_x,v_y = G.nodes[v]['x'],G.nodes[v]['y']
                    killed_resistors_positionsX.append((u_x+v_x)/2.0)
                    killed_resistors_positionsY.append((u_y+v_y)/2.0)
                    G.remove_edge(u,v)
                    nb_Rwire_killed += 1
                    alter_resistances = True

        if alter_resistances == True:
            print("Nb of Rcontact resitors cured: ", nb_Rcontact_cured)
            print("Nb of Rwire resistors killed: ", nb_Rwire_killed)

        if alter_resistances == False:
            print("Aucune Rcontacts dont le courant dépasse IMAX_Rcontacts ET supérieurs à la valeur plancher.")
            print("Fin de l'évolution à tension constante!")
            print("On augmente la tension")
            V_SOURCE_LEVEL += V_SOURCE_LEVEL_INCR






    """
    Comment appeller ce graphe ? à la première itération, je trace
    le potentiel qui correspond à l'état initial, et je plotte aussi
    les résistances qui "vont" être supprimées...
    """                
    
    plot_skip = 5
    if (step == 1) or (step % plot_skip == 0):
        NF.PlotScatterVoltage("XYV.npz",figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}".format(step, alter_mode, nbAlter))
        
        plt.plot(killed_resistors_positionsX,killed_resistors_positionsY,'b.')
        plt.savefig("V_step_{0:04d}.png".format(step), dpi=200)
        plt.close()  
        

    """
    j'enregistre aussi les points correspondants aux résistances tuées
    les N première lignes correspondent aux N premiers steps, uniquement
    si je tue une resistance par step.
    Et encore, pas forcément, car quand j'augmente la tension, je me décale
    d'un step.
    A n'utiliser que comme description cosmétique....
    En fait, le mieux, ce serait d'enregistrer cette liste de points
    comme un attribut du graphe !!!
    
    """

    blue_dots = np.column_stack((killed_resistors_positionsX, killed_resistors_positionsY))
    np.savetxt('blue_dots.txt', blue_dots)
    G.graph['blue_dots']=blue_dots


    """
    NF.PlotScatterCurrent("XYI.npz", figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}".format(step, alter_mode, nbAlter))
    plt.savefig("I_step_{0:04d}.png".format(step), dpi=200)
    plt.close()
    """


    """        
    fig = plt.gcf()
    fig.set_size_inches((8.5,11), forward=True)
    plt.savefig("V_step_{0:04d}.png".format(step), dpi=200)
    plt.close()    
    """
    
   
    if alter_resistances:
        if alter_mode == 'I_Rwires':
            print("{0} -- IMAX_Rwires = {1:.3e} \t i = {2:.3e}".format(step,IMAX_Rwires,i))
        elif alter_mode == 'P_Rwires':
            print("{0} -- PMAX_Rwires = {1:.3e} \t p = {2:.3e}".format(step,PMAX_Rwires,p))
        elif alter_mode == 'Rcontacts':
            print("{0} -- IMAX_Rcontacts = {1:.3e} \t i = {2:.3e}".format(step,IMAX_Rcontacts,i))

 
                         
    fig = plt.gcf()
    #fig.set_size_inches((8.5,11), forward=True)
    #fig.set_size_inches((6.5,8.5), forward=True)
    
    
  





    
    """
    Cleaning
    """
    
    if not(NF.IsGraphPercolating(G)):
        print("Graph is not percolating anymore.  EXIT !!!!")
        print("Vsource_level reached: ", V_SOURCE_LEVEL)

        """
        Le réseau n'est plus percolant. On enregistre une dernière image des potentiels'
        """
        NF.PlotScatterVoltage("XYV.npz",figtitle="step {0:04d}\nalter mode: {1} nb Alter: {2}".format(step, alter_mode, nbAlter))
        
        plt.plot(killed_resistors_positionsX,killed_resistors_positionsY,'b.')
        plt.savefig("V_step_{0:04d}.png".format(step), dpi=200)
        plt.close()  

        break



    NF.RemoveOrphanNodes(G)
    NF.RemoveDanglingNodes(G)

    NF.RemoveIsolatedLoops(G)

    #   Re-indexing nodes before building and solving KCL matrix
    #   NEVER forget ordering='sorted' !!
    G = nx.convert_node_labels_to_integers(G, label_attribute='node_string',
                                           ordering='sorted')


    vecV, matA, vecB = NF.BuildMatrixandSolve(G, Isource_level=I_SOURCE_LEVEL )
    Vplus = vecV[-1]
    
    Rmacro = Vplus / I_SOURCE_LEVEL   #  U =RI
    
    
    """
    Scaling to Vsupply = V_SOURCE_LEVEL
    """
    scaling_factor = V_SOURCE_LEVEL / Vplus
    vecV = vecV * scaling_factor
    I_macro = I_SOURCE_LEVEL * scaling_factor



    
    """
    After solving the KCL matrix, let's update graph nodes with a new
    attribute 'v', equals to the voltage on each node.

    Let's also update edges, by adding two attributes: 'i' (current)
    and 'p' (power)

    Finally, let's add a new graph attribute 'V_source_level'

    Of course, this has to be done each time we solve the KCL matrix !
    """
    for index,v in enumerate(vecV):
        G.nodes[index]['v']=vecV[index]
        
    for u,v,d in G.edges(data=True):
        G[u][v]['i'] = np.abs(G.nodes[u]['v']-G.nodes[v]['v']) / d['Rval']
        G[u][v]['j'] = G[u][v]['i'] / Sw
        G[u][v]['p'] = d['Rval'] * (G[u][v]['i'])**2
    
    G.graph['V_source_level']=V_SOURCE_LEVEL
    G.graph['Rmacro']=Rmacro
    G.graph['step']=step
        
    
    
    
    
    if not NO_SAVE_GRAPH:
        """
        Saving the current graph to disk  (graph,nodes and edges, with all their attributes)
        """
        graph_filename_to_be_zipped = "graph_step_{0:04d}.gpickle".format(step)
        
         ###   DEbugging NAN, avec le defect
        if np.isnan(Rmacro):
            rand_value = random.randint(10,1500)
            shutil.copyfile(graph_filename_to_be_zipped, "graph_evol_debug_{0:04}.gpickle".format(rand_value) )
        
        
        
        archive_filename = os.path.splitext(graph_filename_to_be_zipped)[0]+".zip"
        nx.write_gpickle(G, graph_filename_to_be_zipped)
        myZipFile = zipfile.ZipFile(archive_filename, "w")
        myZipFile.write(graph_filename_to_be_zipped, compress_type=zipfile.ZIP_DEFLATED)
        myZipFile.close()
        os.remove(graph_filename_to_be_zipped)
        
        
        ###   DEbugging NAN, avec le defaut
        if np.isnan(Rmacro):
            rand_value = random.randint(10,1500)
            shutil.copyfile("graph_original.gpickle", "graph_original_{0:04}.gpickle".format(rand_value) )
            shutil.copyfile("original_NWlist.p", "original_NWlist_{0:04}.p".format(rand_value) )
    
    
    
    print("Rmacro :{0:.4f}".format(Rmacro))
    #NF.Record_Rmacros(step,Density, LsX, LsY,Lw, Rlineic_VALUE, Rc_VALUE, Relectrode_VALUE, Rmacro, V_SOURCE_LEVEL)
    NF.Record_Rmacros(step,G.graph['density'], G.graph['LsX'], G.graph['LsY'],G.graph['Lw'], 
                  G.graph['Rlineic'], G.graph['Rc'], G.graph['Relectrode'], G.graph['Rmacro'], 
                  G.graph['V_source_level'])

    #  Save X,Y,Voltages to a .npz file
    NF.SaveScatterVoltages(G)
    NF.SaveScatterCurrents(G)
    
    #shutil.copyfile("XYV.npz", "XYV_{0:>04d}.npz".format(step)) 
    #shutil.copyfile("XYI.npz", "XYI_{0:>04d}.npz".format(step))


    if np.isnan(Rmacro):
        print("Rmacro = NAN.  On sort...")
        break




"""
PLOTTING
"""

"""
fig,ax = plt.subplots()
Rmacros_values = np.loadtxt("Rmacros.txt", skiprows=1, usecols=(9,))
ax.plot(Rmacros_values, marker='*')
ax.set(title="Rmacro = f(step) ")
plt.show()
"""








#if (nb_intersections < 1e5) :
if PLOTTING:
    NF.Plot_Nanowires_List(NWlist, color='black', fig_title="Nanowires List with percolating nodes (dangling nodes removed)\nATTENTION: chaque point blanc cache 2 noeuds, à cause de Rcontacts")

    if percolating:    
        xx = [d['x'] for n,d in G.nodes(data=True)]
        yy = [d['y'] for n,d in G.nodes(data=True)]
        
        xx = []
        yy = []
        for n,d in G.nodes(data=True):
            if d['on_electrode'] == None:
                xx.append(d['x'])
                yy.append(d['y'])
            else:
                if d['on_electrode'].startswith('virtual') == False:
                    xx.append(d['x'])
                    yy.append(d['y'])
                    
                
        
        plt.plot(xx,yy, 'w.', markersize=1.0)
        
    
        #NF.Plot_Nanowires_List(NW_query_list,color='red',overplot=True)
        #plt.plot(x_query,y_query,'ko')
    
        """
        for nano in NWlist:
            label_x = (nano.P1.x + nano.P2.x) / 2.0
            label_y = (nano.P1.y + nano.P2.y) / 2.0
            plt.text(label_x, label_y, '%d' % (nano.index), ha='center', color='blue')
        """
    
        
        """
        D = np.column_stack((X,Y,V))
        np.savetxt('XYV.txt', D)
        (xi,yi,vi) = NF.Interpolate_XYV()
        NF.PlotXiYiVi_PotentialMap(xi,yi,vi)
        """
    
    
    plt.show()



if PROFILING:
    pr.disable()
    pr.dump_stats("profile.out")
    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(.1)