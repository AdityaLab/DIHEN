#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:16:42 2021

@author: anikat
This is the script to check number of spread vary k
"""
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
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

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

def ic_relay(G,seeds,filename,m,pre_computed_p):
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
    #print('chosen seeds:',seeds)
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,pre_computed_p,steps=0)
        #print("IC run:"+str(mc))
        #print(H)
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
    #print('val',val)
    #print('spread ',spread)
    return val,spread


def degreeDiscountIC(path,data,nodes_map,k,p,m):
    G1=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['p','act_prob'],create_using=nx.DiGraph())
    S0=[]
    S0_avg=[]
    d={} #out_degrees
    dd={} #degree_discount
    t={}
    Maxnodes=len(G1.nodes())
    cur=[False]*Maxnodes
    all_nodes=list(G1.nodes())
    for u in all_nodes:
        tmp=0
        for v in G1.out_edges(u):
            if v[1]!=u and float(G1[u][v[1]]['p'])<G1[u][v[1]]['act_prob']: #avoiding self loop conditions
                ##tmp+=G[u][v[1]]['act_prob']
                tmp+=1
        d[u]=tmp
        dd[u]=d[u]
        t[u]=0
    
    num_seeds=0
    node_lookup=0
    while len(S0)<Maxnodes and num_seeds<k:
        s=choose_next_s(all_nodes,dd,S0,cur)
        cur[s[1]]=True
        node_lookup+=1
        S0.append(s[0])
        S0_avg.append(s[2])
        num_seeds+=1
        for child in G1.out_edges(s[0]):
            v=child[1]
            #v!=s[0] is to avoid self loop if present
            if v not in S0 and v!=s[0]:
                if float(G1[s[0]][v]['p'])<G1[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    val,failure_spread=ic_relay(G1, S0, path,m,True)
    strseeds=[nodes_map[s] for s in S0]
    
    return strseeds,S0,failure_spread
    
def get_act_prob(data,pij):
    act_prob=[pij]*data.shape[0]
    
    return act_prob

def collect_seeds(path,G,val,nodes_map,k,m):
    from collections import OrderedDict
    sorted_val = OrderedDict(sorted(val.items(),key=lambda item: item[1],reverse=True))
    
    S0=[]
    num_seed=0
    for key in sorted_val.keys():  
        if num_seed>=k:
            break
        #node_type=nodes_map[key].split(':')[0]
        S0.append(key)
        num_seed+=1
    
    strseeds=[nodes_map[s] for s in S0] 
    
    return strseeds, S0

def page_rank(path,data,nodes_map,k,m,is_weighted=False):
    if is_weighted:
        G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
        pr= nx.pagerank(G, alpha=0.9,weight='act_prob')
    else:
        G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
        pr= nx.pagerank(G, alpha=0.9)    
    
    return collect_seeds(path,G,pr,nodes_map,k,m)

def critical_score(path,data,nodes_map,k,m,pr_score,rpr_score,is_weighted=False):
    if is_weighted:
        G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
        G_rev=nx.DiGraph.reverse(G)
        pr= nx.pagerank(G, alpha=0.9,weight='act_prob')
        r_pr=nx.pagerank(G_rev, alpha=0.9,weight='act_prob')
    else:
        G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
        G_rev=nx.DiGraph.reverse(G)
        pr= nx.pagerank(G, alpha=0.9)    
        r_pr=nx.pagerank(G_rev, alpha=0.9,weight='act_prob')
        
    cr_score={}
    for key in pr.keys():
        cr_score[key]=pr_score*pr[key]+rpr_score*r_pr[key]
        
    
    return collect_seeds(path,G,cr_score,nodes_map,k,m)

def degree_centrality_rank(path,data,nodes_map,k,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
    deg=nx.degree_centrality(G)
    
    return collect_seeds(path,G,deg,nodes_map,k,m)

def clustering(path,data,nodes_map,k,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
    cluster=nx.clustering(G)
    
    return collect_seeds(path,G,cluster,nodes_map,k,m)

def check_baseline_model(filedir,outdir,outfile,K,m,reg):
    path=filedir+'new_preprocessed_data/'
    
    robust_file=open(outdir+outfile+'_spread.csv','w')
    robust_file.writelines('pj,k,lookup,spread\n')
    
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    #nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
    G=nx.from_pandas_edgelist(df, 'u', 'v',edge_attr=['p','act_prob'],create_using=nx.DiGraph())
    P=[1,0.7,0.5,0.2]
    for k in K:
        print('k:',k)
        print('critical score:0,1')
        seed_list,seed_idx=critical_score(path,df,nodes_map,k,m,0,1,is_weighted=False)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='critical_score_0_1'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
        
        print('critical score:1,0')
        seed_list,seed_idx=critical_score(path,df,nodes_map,k,m,1,0,is_weighted=False)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='critical_score_1_0'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
        
        print('critical score:0.5,0.5')
        seed_list,seed_idx=critical_score(path,df,nodes_map,k,m,0.5,0.5,is_weighted=False)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='critical_score_5_5'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
    
        print('page rank unweighted:')
        seed_list,seed_idx=page_rank(path,df,nodes_map,k,m)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='pagerank_unweighted'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
    
        print('page rank weighted:')
        seed_list,seed_idx=page_rank(path,df,nodes_map,k,m,is_weighted=True)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='pagerank_weighted'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
    
        print('degree-centrality:')
        seed_list,seed_idx=degree_centrality_rank(path,df,nodes_map,k,m)
        val,final_spread=ic_relay(G, seed_idx, path,m,True)
        line='degree-cent'+','+str(k)+','+str(len(seed_list))+','+str(final_spread)+'\n'
        robust_file.writelines(line)
        
        for pj in P: 
            tmp_data=df.copy()
            tmp_data['act_prob']=get_act_prob(df,pj)
            
            seed_list,seed_idx,spread = degreeDiscountIC(path,tmp_data,nodes_map,k,pj,m)
            print('diffusion--',pj,spread)
            val,final_spread=ic_relay(G, seed_idx, path,m,True)
            line=str(pj)+','+str(k)+','+str(spread)+','+str(final_spread)+'\n'
            robust_file.writelines(line)
        
        print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
        
        
    robust_file.close()
    

if __name__=='__main__':
    filedir=sys.argv[1] #'data/G_for_robustness/' #for ercot baseline
    outdir='output/baseline_results_spread/'#sys.argv[2]
    outfile=sys.argv[2] #'baseline_for_k'
    m=1
    k=[50,100,200,300,500]#sys.argv[5]
    
    edgefile='national_all_edges_rule_based_pj_p'
    
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    

    reg= sys.argv[3] #'national'
    check_baseline_model(filedir,outdir,outfile,k,m,reg)
    
