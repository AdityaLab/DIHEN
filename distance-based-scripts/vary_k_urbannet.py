#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:13:02 2021

@author: anikat
"""
#import sys
import os
import numpy as np
import networkx as nx
import time
import math
import preprocess_graph as prepg
import csv
import pandas as pd
from independent_cascade import *
import pickle

#sys.stdout = open('simulation_run_urbannet_ddic.txt','w')
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


def evaluate_seeds_1_hop(transmissions,substations,trans_sub,seeds,
                         k,critical_fac,reg,is_voltage_rule,outdir=None):
    
    infile='../data/v9/'
    rules={}
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]
    if outdir is not None:
        selected_seed_data.to_csv(outdir+reg+'_topk_transmission_lines.csv',index=False)
    seed_vol_match1=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    vol_match1=len(seed_vol_match1.drop_duplicates().values)
    seed_vol_match1=seed_vol_match1['NODE_ID']
    
    
    seed_vol_match2=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 230]
    vol_match2=len(seed_vol_match2.drop_duplicates().values)
    seed_vol_match2=seed_vol_match2['NODE_ID']
    
    if is_voltage_rule:
        rules['v>=345']=vol_match1
        #rules['v>=230']=vol_match2
        
    sub_critical=pd.read_csv(infile+'_Electric_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    
    trans_sub=trans_sub[trans_sub['u'].isin(seeds)]
    
    sub_with_seeds=trans_sub['v'].drop_duplicates().values
    num_seed_to_sub=len(sub_with_seeds)
    
    #get all those transmission lines whose substations are connected to military base
    #1. collect all the subs that connected to seeds and military
    seeded_subs_military=sub_critical[sub_critical['u'].isin(sub_with_seeds)]
    #print('#seed-connected substations connected to critical facility:',seeded_subs_military.shape)
    
    seeded_subs_in_military=seeded_subs_military['u'].drop_duplicates().values
    print('#connected critical facility:',len(seeded_subs_in_military))
    #num_sub_to_critical=len(seeded_subs_in_military)
    
    #collect all the transmission lines that are supplying to military
    trans_in_military=trans_sub[trans_sub['v'].isin(seeded_subs_in_military)]
    
    #print('actual critical transmissions',trans_in_military.shape)
    
    trans_in_military=trans_in_military['u'].drop_duplicates().values
    key='near-'+critical_fac
    rules[key]=len(trans_in_military)
    
    true_critical_data=transmissions[transmissions['NODE_ID'].isin(trans_in_military)]
    true_critical_data1=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 345]
    #true_critical_data2=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 230]
    #true_critical_data3=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 138]
    true_critical_nodes=true_critical_data1[['NODE_ID']].values
    rules['v>=345+'+key]=len(true_critical_data1)
    #rules['v>=230+'+key]=len(true_critical_data2)
    #rules['v>=138+'+key]=len(true_critical_data3)
    #print('true nodes',true_critical_nodes.tolist()[0])
    return true_critical_nodes.tolist(),rules,num_seed_to_sub

def degreeDiscountIC(path,data,nodes_transmission,nodes_map,k,p,m):
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
            
    val,failure_spread=ic_relay(G, S0, path,m,True)
    strseeds=[nodes_map[s] for s in S0]
    
    return strseeds,S0,failure_spread,node_lookup
                
def check_k_robustness(filedir,outdir,outfile,K,p,m,critical_fac,reg):
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                       delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    robust_file=open(outdir+outfile+'_vary_k.csv','w')
    robust_file.writelines('reg,spread,k,lookup,v>=345,near_mil,v>=345+near-mil,near_hos,v>=345+near-hos\n')
    
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
            nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
    df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj','p'])
    
    for k in K:
        print(k)
        tmp_data=df.copy()
        
        tmp_data['act_prob']=df.kij*(1-df.pj)
            
        seed_list,seed_idx,final_spread,lookup = degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,k,p,m)
            
        true_seed_c1,rulesc1,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,trans_sub,seed_list,k,critical_fac[0],reg,True,outdir=outdir)
            
        true_seed_c2,rulesc2,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list,k,critical_fac[1],reg,False)
        
        total_critical_seed=true_seed_c1+true_seed_c2
        total_rule=len(total_critical_seed)
            
        print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
        print('total nodes:',len(list(nodes_map.keys())))
        
        string1=reg+','+str(final_spread)+','+str(k)+','+str(lookup)
        string2=''
        string3=''
        for key in rulesc1.keys():
            #print(key,rulesc1[key])
            string2+=','
            string2+=str(rulesc1[key])
        
        for key in rulesc2.keys():
            #print(key,rulesc2[key])
            string3+=','
            string3+=str(rulesc2[key])

        string3+='\n'
        robust_file.writelines(string1+string2+string3)
       
        with open(outdir+outfile+'_true_seed_c1_'+'k_'+str(k)+'.txt','w') as res:
            res.writelines(["%s\n" % item  for item in true_seed_c1])
            
        with open(outdir+outfile+'_true_seed_c2_'+'k_'+str(k)+'.txt','w') as res:
            res.writelines(["%s\n" % item  for item in true_seed_c2])
        
            
    robust_file.close()
        
    
if __name__=='__main__':
    filedir= sys.argv[1] #'data/'
    outfile=sys.argv[2] #'vary_k_national'
    reg=sys.argv[3] #'EIC','TX','national'
    outdir= 'output/G_vary_k_urbannet/'#sys.argv[2]
    m=10 #number of iteration in ic
    K=[25,50,100,200,300,500]
    
    p=0.1 
    edgefile='all_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases','Hospitals']
    
    check_k_robustness(filedir,outdir,outfile,K,p,m,critical_fac,reg)
