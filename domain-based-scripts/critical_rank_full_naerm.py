#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 3 13:13:02 2021

@author: anikat
"""
import sys
import os
import numpy as np
import networkx as nx
import time
import math
import preprocess_full_urbannet as prepg
import csv
import pandas as pd
from independent_cascade import *
import pickle

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    return arrayt

def draw_plot_k(k,spread):
    import matplotlib.pyplot as plt
    plt.plot(k,spread)
    plt.show()
    
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

def ic_relay(G,seeds,filename,m,alpha=0.2,pre_computed_p=False):
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
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,alpha,pre_computed_p,steps=0)
        print("IC run:"+str(mc))
        for i in range(0,len(H)):
            for j in range(0,len(H[i])):
                if H[i][j] in list(G.nodes):
                    val[H[i][j]]+=1
    for i in val.keys():
        #val[i]=1-val[i]/m
        val[i]=val[i]/m
        spread+=val[i]
    return val,spread
            
def degreeDiscountIC(indir,outdir,outfile,edgefile,k,p,m=10):
    print('Begin:')
    start_time = time.perf_counter()
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes.txt')
    df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
    #df['act_prob']=df.kij
    G=nx.from_pandas_edgelist(df,'u','v',edge_attr=['act_prob','p'],create_using=nx.DiGraph())
    
    print('Total nodes:',len(G.nodes()))
    print('Total edges:',len(G.edges()))
     
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
        #print('edges %d:%d'%(u,len(list(G.out_edges(u)))))
        for v in G.out_edges(u):
            #print(v[0],v[1])
            if v[1]!=u and float(G[u][v[1]]['p'])<G[u][v[1]]['act_prob']: #avoiding self loop conditions
                ##tmp+=G[u][v[1]]['act_prob']
                tmp+=1
        d[u]=tmp
        dd[u]=d[u]
        t[u]=0
    
    print("Running time precomputation:--- %s seconds ---" % (time.perf_counter() - start_time))
    num_seeds=0
    node_lookup=0
    while len(S0)<Maxnodes and num_seeds<k:
        s=choose_next_s(all_nodes,dd,S0,cur)
        cur[s[1]]=True
        out_deg=len(list(G.out_edges(s[0])))
        node_lookup+=1
        #print('top deg-ic:',s[0],s[1],s[2])
        #if s[0] in nodes_transmission:
        S0.append(s[0])
        S0_avg.append(s[2])
        num_seeds+=1
        print("chosen %d th node=%d; ddv[%d]=%f, out_edges=%f" % (num_seeds,s[0],s[1],s[2],out_deg))
        for child in G.out_edges(s[0]):
            #print(v[0],v[1],v[2])
            v=child[1]
            if v not in S0 and v!=s[0]: #v!=s[0] is to avoid self loop if present
                if float(G[s[0]][v]['p'])<G[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    print("The final best k critical transmission nodes after %d search are:"%node_lookup)    
    print("Running time to find k critical nodes:--- %s seconds ---" % (time.perf_counter() - start_time))
    strseeds=[nodes_map[s] for s in S0]
    
    result=open(outdir+outfile+'_results_seeds.txt','w')
    result.writelines(["%s\n" % item  for item in strseeds])
    result.close()
    
    result2=open(outdir+outfile+'_results_idx_seeds.txt','w')
    result2.writelines(["%s\n" % str(item)  for item in S0])
    result2.close()
    '''
    S0=[]
    with open(outdir+outfile+'_results_idx_seeds_'+str(k)+'.txt', 'r') as res:
        for line in res: 
            line = line.strip('\n') #or some other preprocessing
            S0.append(int(line)) 
    '''
    '''
    #topK=[500]
    for k in topK:          
        val,failure=ic_relay(G, S0[:k], path,m,pre_computed_p=True)
        print('final ic failures given k= %d : %f' %(k,failure))
        print("IC running time:--- %s seconds ---" % (time.perf_counter() - start_time))
        strfailure=[]
        for n in val.keys():
            if val[n]>0:
                strfailure.append(nodes_map[n])
        print(strfailure[0],len(strfailure))
        with open(outdir+outfile+'_ic_failure_top_'+str(k)+'.txt', 'w') as f:
            for item in strfailure:
                f.writelines("%s\n"%item)
            #pickle.dump(val, f)
    '''    
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))

if __name__=='__main__':
    filedir='data/naerm/'
    
    outdir='output/naerm/'#sys.argv[3]
    outfile='naerm_allnodes_' #sys.argv[4] 
    m=1#sys.argv[5]
    k=int(sys.argv[1])#303546
    
    p=0.01 #sys.argv[7]
    
    edgefile='naerm_edges_rule_based_pj_p' #sys.argv[8]
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
      
    degreeDiscountIC(filedir,outdir,outfile,edgefile,k,p,m)
    
        
