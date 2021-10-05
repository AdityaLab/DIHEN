#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 06:00:08 2021

@author: anikat
"""
import sys
import os
import numpy as np
import networkx as nx
import time
import csv
import pandas as pd
from independent_cascade import *

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    
    return arrayt
            
def choose_next_s(nodes,delta,S0,cur):
    '''
    this part is to choose the node with max degree-discount gain
    '''
    #set min marginal gain be a smaller number, so any results will be chosen to update mind
    maxddv=-10000  
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
    
    for i in val.keys():
        val[i]=val[i]/m
        spread+=val[i]
    
    return val,spread

def degreeDiscountIC(path,data,nodes_transmission,nodes_map,k,p,m,return_cascade=False):
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
        node_lookup+=1
        S0.append(s[0])
        S0_avg.append(s[2])
        
        num_seeds+=1
        
        for child in G.out_edges(s[0]):
            v=child[1]
            #v!=s[0] is to avoid self loop if present
            if v not in S0 and v!=s[0]:
                if float(G[s[0]][v]['p'])<G[s[0]][v]['act_prob']: 
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    val,failure_spread=ic_relay(G, S0, path,m,False)
    strseeds=[nodes_map[s] for s in S0]
    '''
    strcascade=[nodes_map[s] for s in cascade_l]
    if return_cascade:
        return strcascade,cascade_l,failure_spread,node_lookup
    '''
    return strseeds,S0,failure_spread,node_lookup

def check_model_spread(filedir,outdir,outfile,edgefile,reg,K,p,m):
    spread_file=open(outdir+outfile+'.csv','w')
    spread_file.writelines('node_id,node_name,DIHEN,spread\n')
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
    
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p']) 
    df['act_prob']=df.kij*(1-df.pj)
    '''
    seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(filedir,df,nodes_transmission,nodes_map,K[0],p,m)
    
    
    with open(outdir+'naerm_seeds_'+str(K[0])+'.txt', 'w') as f:
        for idx in range(0,len(seed_list)):
            f.write("%s,%d\n" % (seed_list[idx],seed_idx[idx]))
    '''
    with open(outdir+'naerm_seeds_'+str(K[0])+'.txt', 'r') as f:
        seed_idx = []
        for line in f:
            nodes=line.split(',')
            #print(nodes[0],int(nodes[1]))
            seed_idx.append(int(nodes[1]))
    
    
    G=nx.from_pandas_edgelist(df, 'u', 'v',edge_attr=['p','act_prob'],create_using=nx.DiGraph())
    for nid in nodes_map.keys():
        seed=[nid]
        val,failure_spread=ic_relay(G, seed, filedir,m,False)
        dihen='N'
        if nid in seed_idx:
            dihen='Y'
        
        print(nid,failure_spread)
        string=str(nid)+','+nodes_map[nid]+','+dihen+','+str(failure_spread)+'\n'
        spread_file.writelines(string)
    
    spread_file.close()
    
if __name__=='__main__':
    filedir='../data/naerm/naerm_regional/' #sys.argv[1]
    outdir='../output/naerm/'#sys.argv[2]
   
    outfile='naerm_empirical_spread' #sys.argv[3] 
    K=[500]
    m=10  
    p=0.1 
    edgefile='national_naerm_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    region='national'
    
    check_model_spread(filedir,outdir,outfile,edgefile,region,K,p,m)

