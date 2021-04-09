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

sys.stdout = open('simulation_run_all_combination.txt','w')
def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt


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
    
    
    Maxnodes=len(nodes_transmission)
    delta=[Maxnodes+1]*(Maxnodes)
    max_failure=0
    best_nodes=[]
    for i in range(len(nodes_transmission)):
        for j in range(i+1,len(nodes_transmission)):
            S1=[nodes_transmission[i],nodes_transmission[j]]
            val1,failure=ic_relay(G, S1, path,alpha,m)
            print("Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
            print("node=[%d, %d]; spread=%f" % (S1[0],S1[1],failure))
            if failure>max_failure:
                max_failure,best_nodes=failure,S1
                
    
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
    strseeds=[nodes_map[s] for s in best_nodes]
    print("The final best k critical transmission nodes are:")
    print(strseeds,best_nodes)
    print('Total failures in the network given S0:',max_failure)
    result=open(outdir+outfile+'_all_comb_ic_seeds_'+str(k)+'.txt','w')
    result.writelines(["%s," % item  for item in strseeds])
    result.writelines("\n")
    result.writelines(["%s," % max_failure])
    
if __name__=='__main__':
    filedir='data/toy/' #sys.argv[1]
    consumer1='Natural_Gas_Compressor_Stations'#sys.argv[2]
    #read_whole_graph(filedir,consumer1)
    m=5#sys.argv[3]
    k=2#sys.argv[4]
    #alpha: hyperparam choose stress increase on a node when its co-parent fails
    alpha=0.2 #sys.argv[5]  
    outdir='output/'#sys.argv[5]
    outfile='toy'
    #outfile='urbannet' 
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    greedy(filedir,outdir,outfile,k,consumer1,alpha,m)
    #seeds=[5,7]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
    #seeds=[11,13]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
        