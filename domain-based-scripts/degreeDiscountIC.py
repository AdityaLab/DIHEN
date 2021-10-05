#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:53:29 2021

@author: anikat
"""

import networkx as nx
#import pandas as pd
from independent_cascade import independent_cascade

def ic_relay(G,seeds,filename,m,pre_computed_p):
    #S: failed seed nodes that have been selected so far
    #m: # simulation to run
    
    spread = 0
    Maxnodes=max(list(G.nodes))
    val={}
    for node in list(G.nodes):
        val[node]=0
    val[0]=0
    test=0
    
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,pre_computed_p,steps=0)
    
        for i in range(0,len(H)):
            for j in range(0,len(H[i])):
                if H[i][j] in list(G.nodes):
                    val[H[i][j]]+=1
    
    cascade_list=[]
    for i in val.keys():
        if val[i]>0:
            cascade_list.append(i)
        val[i]=val[i]/m
        spread+=val[i]
    
    return val,cascade_list,spread

def choose_next_s(nodes,delta,S0,cur):
    '''
    this part is to choose the node with max degree-discount gain
    '''
    #set min marginal gain be a smaller number, so any results will be chosen to update mind
    maxddv=-1000  
    for i in range(0,len(nodes)):
        #print("i=%d" % i)
        if i==0 and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
        elif delta[nodes[i]] > maxddv and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
    #print("mins=%d,i=%d" % (mins,mini))
    return [ddv,maxi,maxddv]

def degreeDiscountIC(path,data,nodes_transmission,nodes_map,k,p,m,pre_computed_p,return_cascade=False):
    G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['p','act_prob'],create_using=nx.DiGraph())
    S0=[]
    S0_avg=[]
    d={} #out_degrees
    dd={} #degree_discount
    t={}
    Maxnodes=len(G.nodes())
    Max_trans_nodes=len(nodes_transmission)
    cur=[False]*Maxnodes
    all_nodes=list(G.nodes())
    for u in all_nodes:
        tmp=0
        for v in G.out_edges(u):
            if v[1]!=u and float(G[u][v[1]]['p'])<G[u][v[1]]['act_prob']: #avoiding self loop conditions
                ##tmp+=G[u][v[1]]['act_prob']
                tmp+=1
        d[u]=tmp
        dd[u]=d[u]
        t[u]=0
    
    num_seeds=0
    node_lookup=0
    while len(S0)<Max_trans_nodes and num_seeds<k:
        s=choose_next_s(all_nodes,dd,S0,cur)
        cur[s[1]]=True
        #out_deg=len(list(G.out_edges(s[0])))
        node_lookup+=1
        #print('top deg-ic:',s[0],s[1],s[2])
        
        if s[0] in nodes_transmission:
            S0.append(s[0])
            S0_avg.append(s[2])
            #num_seeds+=1
        
        num_seeds+=1
            #print("chosen %d th node=%d; ddv[%d]=%f, out_edges=%f" % (num_seeds,s[0],s[1],s[2],out_deg))
        for child in G.out_edges(s[0]):
            v=child[1]
            #v!=s[0] is to avoid self loop if present
            if v not in S0 and v!=s[0]:
                if float(G[s[0]][v]['p'])<G[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    val,cascade_l,failure_spread=ic_relay(G, S0, path,m,pre_computed_p)
    strseeds=[nodes_map[s] for s in S0]
    strcascade=[nodes_map[s] for s in cascade_l]
    if return_cascade:
        return strcascade,cascade_l,failure_spread,node_lookup
    
    return strseeds,S0,failure_spread,node_lookup
