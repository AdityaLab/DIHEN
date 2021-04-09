#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:35:11 2021

Adapted Independent Cascade model to work for model-relay

Implement independent cascade model
"""
#!/usr/bin/env python
#    Copyright (C) 2004-2010 by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/.
__author__ = """Hung-Hsuan Chen (hhchen@psu.edu)"""

import copy
import networkx as nx
import random
import time
import numpy as np

def init_graph(G,seeds,filename):
    lines=[]
    kk=0
    lineNum=0
    f=open(filename+'all_edges.txt','r')
    for line in f:
        oneline=line.strip('\n').split(',')
        lineNum+=1
        lines.append(oneline)
        #G.add_edge(int(lines[kk][0]),int(lines[kk][1]))
        kk+=1
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("independent_cascade() is not defined for graphs with multiedges.")

    # make sure the seeds are in the graph
    for s in seeds:
        if s not in list(G.nodes):
            raise Exception("seed", s, "is not in graph")
  
    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)
    #record num of incoming edges failing for each node, i.e., n (\alpha)^n
    num_failed_parents={}
    for n in list(DG.nodes):
        num_failed_parents[n]=0
    
    nodes_pj={}
    # init activation probabilities
    num_edges=0
    for e in DG.edges():
        #storing num of failed parents for each child to compute (alpha)^n
        if e[0] in seeds:
          num_failed_parents[e[1]]+=1  
        #if typeN == 1:
        #print(["edges=",e[0],e[1]])
        print('edge-num',num_edges)
        if 'act_prob' not in DG[e[0]][e[1]]:
            flag=0
            for tuple in lines:
                #print(["tuple",tuple[0],tuple[1],tuple[2],tuple[3]])
                if (int(tuple[0]) ==e[0] and int(tuple[1]) ==e[1])or(int(tuple[1]) ==e[0] and int(tuple[0]) ==e[1]):
                    flag=1
                    DG[e[0]][e[1]]['act_prob']=float(tuple[2])
                    if e[1] not in nodes_pj.keys():
                        nodes_pj[e[1]]=float(tuple[3])
            if flag==0:
                DG[e[0]][e[1]]['act_prob']=0.0
            #print(["Edge not found!", type(e[0]),len(DG.edges()),lineNum])
            #print("e0=%s,e1=%s,prob=%f"% (e[0],e[1],DG[e[0]][e[1]]['act_prob']))

    
        elif DG[e[0]][e[1]]['act_prob'] > 1:
            raise Exception("edge activation probability:", \
            DG[e[0]][e[1]]['act_prob'], "cannot be larger than 1")
        num_edges+=1
    return DG,nodes_pj,num_failed_parents
    
    
def independent_cascade(G,seeds,filename,alpha=0.1,steps=0):
    #print(["IN IC",len(G.edges())])
    DG,nodes_pj,num_failed_parents=init_graph(G,seeds,filename)
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)
    
    A = copy.deepcopy(seeds)  # prevent side effect
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(DG, A,nodes_pj,num_failed_parents,alpha) 
    # perform diffusion for at most "steps" rounds
    return _diffuse_k_rounds(DG,A,nodes_pj,num_failed_parents,alpha,steps)

def _diffuse_all(G,A,nodes_pj,num_failed_parents,alpha):
  print("diffuse all")
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])  # prevent side effect
  while True:
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G,A,tried_edges,nodes_pj,num_failed_parents,alpha)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
          #print("2")
  return layer_i_nodes

def _diffuse_k_rounds(G,A,nodes_pj,num_failed_parents,alpha,steps):
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G, A, tried_edges)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
    steps -= 1
        #print("3")
  return layer_i_nodes

def _diffuse_one_round(G,A,tried_edges,nodes_pj,num_failed_parents,alpha):
  activated_nodes_of_this_round = set()
  cur_tried_edges = set()
  for s in A:
    for nb in G.neighbors(s):
      if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
        continue
      
      if _prop_success(G, s, nb,nodes_pj[nb],num_failed_parents[nb],alpha):
        activated_nodes_of_this_round.add(nb)
        children=list(G.out_edges(nb))
        print('added:',nb,len(children))
        for c in children:
            num_failed_parents[c[1]]+=1
      cur_tried_edges.add((s, nb))
  activated_nodes_of_this_round = list(activated_nodes_of_this_round)
  A.extend(activated_nodes_of_this_round)
  
  return A, activated_nodes_of_this_round, cur_tried_edges

def _prop_success(G, src, dest, pj, n, alpha):
  #print("5")
  pij=G[src][dest]['act_prob']*(1-np.power(alpha,n-1)*pj)
  if pij > 1 or pij<0:
      raise Exception("edge activation probability:", \
        pij, pj,n,G[src][dest]['act_prob'],\
            "cannot be larger than 1 or negative")
  
  return random.random() <= pij

def _IC_overload(e1,e2):
    with open('FL_data_result/FL_overload_number.txt','rU') as f:
        for line in f:
            oneline=line.split()
            if (oneline[0] == e1 and oneline[1] == e2) or (oneline[0] == e2 and oneline[1] == e1):
                print("find edge=[%d,%d,%d]" % (oneline[0],oneline[1],oneline[2]))
                return oneline[2]
        print("NOT find edge=[%d,%d]" % (e1,e2))
        return(0)



