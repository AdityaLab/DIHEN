#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:51:29 2021

@author: anikat
This script is to get results for all baselines on domain-based national network
"""
#import sys
import os
import numpy as np
import networkx as nx
import time
import math
import csv
import pandas as pd
from degreeDiscountIC import *
import naerm_critical as CR

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    
    return arrayt

def collect_seeds(path,G,val,nodes_map,k,m):
    from collections import OrderedDict
    sorted_val = OrderedDict(sorted(val.items(),key=lambda item: item[1],reverse=True))
    
    S0=[]
    num_seed=0
    for key in sorted_val.keys():  
        if num_seed>=k:
            break
        node_type=nodes_map[key].split(':')[0]
        if node_type=='Transmission_Lines':
            #print(key,sorted_val[key])
            S0.append(key)
        num_seed+=1
    
    strseeds=[nodes_map[s] for s in S0] 
    #val,failure_spread=ic_relay(G, S0, path,m,True)
    
    return strseeds, S0, -1

def get_act_prob(data,pij):
    act_prob=[pij]*data.shape[0]
    
    return act_prob

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

def writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,
              seed_list,k,model_name,critical_fac,lookup,final_spread):
    
    original_R7,R7 = CR.evaluate_R7(trans_sub,dis_sub,seed_list,k,nodes_map)
            
    original_R8,R8=CR.evaluate_R8(trans_sub,dis_sub,trans_to_dis,seed_list,nodes_map,k,critical_fac[0])
            
    original_R1, R1 = CR.evaluate_R1(trans_sub,trans_line,seed_list,nodes_map,k)
            
    R2,R3=CR.evaluate_R2R3(dis_sub,trans_sub,seed_list,k,critical_fac[1])
    original_R9,R9 = -1,-1
    
    print(R1,R2,R3,R7,R8,R9)
            
    string1=model_name+','+str(len(seed_list))+','+str(final_spread)+','+str(lookup)+','
    string2=str(R1)+','+str(R2)+','+str(R3)+','
    string3=str(R7)+','+str(R8)+','+str(R9)+'\n'
    
    return string1+string2+string3

def check_baseline_model(path,outdir,outfile,k,p,m,P,critical_fac,reg):
    infile='../data/naerm/edge-files/'
    trans_line=pd.read_csv(infile+'_Transmission_Lines.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    dis_sub=pd.read_csv(infile+'_Distribution_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_to_dis=pd.read_csv(infile+'_Transmission_Substations-_Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    baseline_file=open(outdir+outfile+'.csv','w')
    baseline_file.writelines('model,k,spread,lookup,R1,R2,R3,R7,R8,R9\n')
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(path+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes_'+reg+'.txt')
    
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    
    df['act_prob']=df.kij*(1-df.pj)
    
    print('critical score-10:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,1,0,is_weighted=False)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'critical-score10',critical_fac,len(seed_list),final_spread)
    baseline_file.writelines(line)
    
    print('critical score-01:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,0,1,is_weighted=False)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'critical-score10',critical_fac,len(seed_list),final_spread)
    baseline_file.writelines(line)
    
    print('critical score-55:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,1,0,is_weighted=False)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'critical-score10',critical_fac,len(seed_list),final_spread)
    baseline_file.writelines(line)
    
    print('page rank unweighted:')
    seed_list,seed_idx,final_spread=page_rank(path,df,nodes_map,k,m)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'pagerank_noweight',critical_fac,-1,final_spread)
    baseline_file.writelines(line)
    
    print('page rank weighted:')
    seed_list,seed_idx,final_spread=page_rank(path,df,nodes_map,k,m,is_weighted=True)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'pagerank_weighted',critical_fac,-1,final_spread)
    baseline_file.writelines(line)
    
    print('degree-centrality:')
    seed_list,seed_idx,final_spread=degree_centrality_rank(path,df,nodes_map,k,m)
    line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,'deg-cent',critical_fac,-1,final_spread)
    baseline_file.writelines(line)
    
    for pj in P: 
        print('diffusion--',pj)
        tmp_data=df.copy()
        tmp_data['act_prob']=get_act_prob(df,pj)
            
        seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,k,pj,m,False)
        model_name='IC_'+str(pj)
        line=writefile(outdir,outfile,nodes_map,trans_line,dis_sub,trans_sub,trans_to_dis,seed_list,k,model_name,critical_fac,lookup,final_spread)
        baseline_file.writelines(line)
    
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    baseline_file.close()
    
        
    
if __name__=='__main__':
    filedir='../data/naerm/naerm_regional/' #sys.argv[1]
    outdir='../output/naerm/'#sys.argv[2]
   
    outfile='naerm_baseline500_national' #sys.argv[3] 
    m=10
    k=500#sys.argv[4]
    
    p=0.1 
    edgefile='national_naerm_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Compressor_Stations','Military_Bases']
    
    P=[1,0.9,0.7,0.5,0.3,0.2]
    
    reg='national'
    check_baseline_model(filedir,outdir,outfile,k,p,m,P,critical_fac,reg)
    
    
    
