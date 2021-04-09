#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:13:02 2021

@author: anikat
"""
import sys
import os
import numpy as np
import networkx as nx
import time
import math
import preprocess_graph as prepg
import csv
import pandas as pd
from independent_cascade import *

sys.stdout = open('simulation_run_urbannet.txt','w')
def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def choose_next_s(nodes,delta,S0):
    '''
    this part is for lazy evaluation CELF in greedy
    '''
    #set min marginal gain be a smaller number, so any results will be chosen to update mind
    mind=-1000  
    for i in range(0,len(nodes)):
        #print("i=%d" % i)
        if i==0 and nodes[i] not in S0:
            mind=delta[i]
            mins=nodes[i]
            mini=i
        elif delta[i] > mind and nodes[i] not in S0:
            mind=delta[i]
            mins=nodes[i]
            mini=i
    #print("mins=%d,i=%d" % (mins,mini))
    return [mins,mini,mind]

def ic_relay(G,seeds,filename,alpha,m):
    #S: failed seed nodes that have been selected so far
    #alpha: hyperparamaeter to select stress increase on uncertainty
    #m: # simulation to run
    
    spread = 0
    Maxnodes=max(list(G.nodes))
    #val=[0]*(Maxnodes+1)
    val={}
    for node in list(G.nodes):
        val[node]=0
    val[0]=0
    test=0
    print('chosen seeds:',seeds)
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,alpha,steps=0)
        print("IC run:"+str(mc))
        print(H)
        #time.sleep(5)
        #__all__ = ['independent_cascade']
        for i in range(0,len(H)):
            for j in range(0,len(H[i])):
                if H[i][j] in list(G.nodes):
                    val[H[i][j]]+=1
    for i in val.keys():
        #val[i]=1-val[i]/m
        val[i]=val[i]/m
        spread+=val[i]
    print('val',val)
    print('spread ',spread)
    return val,spread

def force_seed(indir,outdir,k,consumer1,seeds,alpha=0.2,m=10):
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    
    df = pd.read_csv(path+'all_edges.txt',delimiter=',',names=['u','v','kij','pj'])
    G=nx.from_pandas_edgelist(df, 'u', 'v',create_using=nx.DiGraph())
    val,failure=ic_relay(G, seeds, path,alpha,m)
    
    strseeds=[nodes_map[s] for s in seeds]
    print("Chosen seeds:",strseeds)
    print("spread:",failure)
    
    
def greedy(indir,outdir,outfile,k,consumer1,alpha=0.2,m=10):
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes.txt')
    df = pd.read_csv(path+'all_edges.txt',delimiter=',',names=['u','v','kij','pj'])
    df['act_prob']=df.kij*(1-df.pj)
    G=nx.from_pandas_edgelist(df, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
    
    print('Total nodes:',len(G.nodes()))
    print('Total edges:',len(G.edges()))
    
    S0=[]
    S0_avg=[]
    look_ups=[]
    Maxnodes=len(nodes_transmission)
    delta=[Maxnodes+1]*(Maxnodes)
    cur=[False]*(Maxnodes)
    val_2={}
    for node in list(G.nodes):
        val_2[node]=1
    val_2[0]=1
    failure=0
    while (len(S0) < Maxnodes) and len(S0) < k:
        for i in range(0,Maxnodes):
            cur[i]=False
        node_lookup=0
        while True:
        #for ti in nodes_transmission:
            s=choose_next_s(nodes_transmission,delta,S0)
            if cur[s[1]]: #if current is the top node add current node and its spread in seed
                S0.append(s[0]) 
                S0_avg.append(s[2])
                failure+=s[2]
                look_ups.append(node_lookup)
                print("quit!",node_lookup)
                break
            else:
                #if ti in S0:
                    #   continue
                S1=[s[0]]
                val1,failure1=ic_relay(G, S0+S1, path,m)
                #print('selected:',S1,failure)
                delta[s[1]]=failure1-failure
                cur[s[1]]=True
                print("Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
                print("node=%d; delta[%d]=%f" % (s[0],s[1],delta[s[1]]))
                print(S0)
                print(S0_avg)
                node_lookup+=1
                #if failure>max_failure:
                 #   max_failure,best_node=failure,ti
                
        #print('choosen node and spread:',best_node,max_failure)
        #S0.append(best_node)
        #S0_avg.append(max_failure)
        
    
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
    strseeds=[nodes_map[s] for s in S0]
    print("The final best k critical transmission nodes are:")
    print(strseeds,S0)
    print('Total failures in the network given S0:',S0_avg)
    result=open(outdir+outfile+'_results_ic_seeds_'+str(k)+'.txt','w')
    result.writelines(["%s," % item  for item in strseeds])
    result.writelines("\n")
    result.writelines(["%s," % str(item)  for item in S0_avg])
    
if __name__=='__main__':
    filedir='data/' #sys.argv[1]
    consumer1='Natural_Gas_Compressor_Stations'#sys.argv[2]
    #read_whole_graph(filedir,consumer1)
    m=5#sys.argv[3]
    k=2#sys.argv[4]
    #alpha: hyperparam choose stress increase on a node when its co-parent fails
    alpha=0.2 #sys.argv[5]  
    outdir='output/'#sys.argv[5]
    #outfile='toy'
    outfile='urbannet_no_consumer' 
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    greedy(filedir,outdir,outfile,k,consumer1,alpha,m)
    #seeds=[5,7]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
    #seeds=[11,13]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
        