#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:59:24 2021

@author: anikat
"""
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

#sys.stdout = open('simulation_run_urbannet_ddic.txt','w')
def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def evaluate_seeds_1_hop(transmissions,substations,trans_sub,seeds,
                         k,critical_fac,is_voltage_rule,outdir=None):
    
    infile='data/v9/'
    rules={}
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]   
    if outdir is not None:
        selected_seed_data.to_csv(outdir+'_topk_transmission_lines.csv',index=False)
    seed_vol_match1=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    vol_match1=len(seed_vol_match1.drop_duplicates().values)
    seed_vol_match1=seed_vol_match1['NODE_ID']
    
    
    seed_vol_match2=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 230]
    #vol_match2=len(seed_vol_match2.drop_duplicates().values)
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

def writelinetofile(rulesc1,rulesc2,k,seed_list,score):
    string1=score+','+str(k)+','+str(len(seed_list))
    string2=''
    string3=''
    for key in rulesc1.keys():
        print(key,rulesc1[key])
        string2+=','
        string2+=str(rulesc1[key])
    
    for key in rulesc2.keys():
        print(key,rulesc2[key])
        string3+=','
        string3+=str(rulesc2[key])

    string3+='\n'
    
    return string1+string2+string3        
        
def get_seed_list(nodes_map,node_list,k,es_to_tl_map):
    seed_list=[]
    num_tl=0
    for n in range(k):
       node=nodes_map[node_list[n]]
       if node.split(':')[0]=='Transmission_Lines':
           seed_list.append(node)
           num_tl+=1
       elif node.split(':')[0]=='Electric_Substations':
           seed_list.append(es_to_tl_map[node])

    print('transmission lines found:',num_tl)    
    return seed_list
       
def analyze_page_rank_vs_ic(outfile,outdir,K,critical_fac):
    infile='data/v9/'
    nodes_file=open(infile+'urbannet2020-graph-v9/index_seq.txt','r')
    nodes_list=csv.reader(nodes_file,delimiter=' ')
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    
    crscore_file1=open(outdir+'sort0-10.txt','r')
    crscore_file2=open(outdir+'sort5-5.txt','r')
    score_list1=csv.reader(crscore_file1,delimiter=' ')
    score_list2=csv.reader(crscore_file2,delimiter=' ')
    
    seeds1=[int(row[0]) for row in score_list1]
    seeds2=[int(row[0]) for row in score_list2]
    seeds3=_get_array(outdir+'urbannetv2_kij__results_idx_seeds_3000.txt')
    
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
        
    
    es_to_tl_map={}
    for ix,row in trans_sub.iterrows():
        if row['v'] not in es_to_tl_map.keys():
            es_to_tl_map[row['v']]=row['u']
        else:
            print('substation has >1 tl')
    
    urbannet_file=open(outdir+outfile+'.csv','w')
    urbannet_file.writelines('score,top_k,lookup,v>=345,near_mil,v>=345+near-mil,near_hos,v>=345+near-hos,total_rule\n')
    for k in K:
        seed_list1=get_seed_list(nodes_map,seeds1,k,es_to_tl_map)
        seed_list2=get_seed_list(nodes_map,seeds2,k,es_to_tl_map)
        seed_list3=get_seed_list(nodes_map,seeds3,k,es_to_tl_map)
    
        print(k)
        print('critical-score1 (0-10) analysis:',len(seed_list1))
        true_seed_c11,rulesc11,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list1,k,critical_fac[0],True,outdir=outdir)
            
        true_seed_c21,rulesc21,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list1,k,critical_fac[1],False)
            
        total_critical_seed1=true_seed_c11+true_seed_c21
        #total_critical_seed1=set(total_critical_seed)
        rulesc21['total_rule']=len(total_critical_seed1)
        #print(rulesc11)
        #print(rulesc21)
        
        print('critical-score2 (5-5) analysis:',len(seed_list2))
        true_seed_c12,rulesc12,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list2,k,critical_fac[0],True,outdir=outdir)
            
        true_seed_c22,rulesc22,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list2,k,critical_fac[1],False)
            
        total_critical_seed2=true_seed_c12+true_seed_c22
        
        rulesc22['total_rule']=len(total_critical_seed2)
        #print(rulesc12)
        #print(rulesc22)
    
        print('ic-model analysis:',len(seed_list3))
        true_seed_c13,rulesc13,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list3,k,critical_fac[0],True)
            
        true_seed_c23,rulesc23,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list3,k,critical_fac[1],False)
            
        total_critical_seed3=true_seed_c13+true_seed_c23
        rulesc23['total_rule']=len(total_critical_seed3)
        #print(rulesc13)
        #print(rulesc23)
        
        line1=writelinetofile(rulesc11,rulesc21,k,seed_list1,'critical_score0-10')
        line2=writelinetofile(rulesc12,rulesc22,k,seed_list2,'critical_score5-5')
        line3=writelinetofile(rulesc13,rulesc23,k,seed_list3,'ic-model')
        urbannet_file.writelines(line1)
        urbannet_file.writelines(line2)
        urbannet_file.writelines(line3)
    urbannet_file.close()

if __name__=='__main__':
    filedir='data/G_for_robustness/' #sys.argv[1]
    outdir='output/urbannet-results-top-2500/'#sys.argv[2]
    outfile='compare-results-v2_kij' #sys.argv[3] 
    
    K=[50,100,200,300,400,500,1000]#sys.argv[5]
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases','Hospitals']

    analyze_page_rank_vs_ic(outfile,outdir,K,critical_fac)   


