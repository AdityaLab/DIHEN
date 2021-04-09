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
import preprocess_graph as prepg
import csv
import pandas as pd
from degreeDiscountIC import degreeDiscountIC
import pickle

def get_original_tl_ids(nodes_map):
    tl_list=[]
    for key in nodes_map:
        r=nodes_map[key]
        if r.split(':')[0]=='Transmission_Lines':
            tl_list.append(r)
    
    print('ttl transmission',len(tl_list))
    return tl_list

def get_random_tl_ids(nodes_map,k):
    node_list=nodes_map.keys()
    import random
    random_nodes=random.sample(node_list,k)
    
    tl_list=[]
    for r in random_nodes:
        node=nodes_map[r]
        if node.split(':')[0]=='Transmission_Lines':
            tl_list.append(node)
    
    print('randomly selected tl line:',len(tl_list))
    return tl_list
    
def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def evaluate_connected_tes(outdir,transmissions,trans_es,seeds,k,reg):
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]
    
    filename1='_Transmission_Lines-Transmission_Electric_Substations.edge'
    #filename2='_Transmission_Electric_Substations-Transmission_Lines.edge'
    edge_tline_tes=pd.read_csv('data/'+filename1,delimiter=',',header=None,names=['u','v'])
    
    tline_tes=edge_tline_tes[edge_tline_tes['u'].isin(seeds)]
    
    critical_tes=tline_tes['v'].drop_duplicates().values
    tes_data=trans_es[trans_es['NODE_ID'].isin(critical_tes)]
    tes_138_data=tes_data.loc[tes_data['NOM_KV']>=138]
    tes_138=tes_138_data['NODE_ID'].values
    
    seed_345_data=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    seed_345=seed_345_data['NODE_ID'].values
    tline_tes_345=edge_tline_tes[edge_tline_tes['u'].isin(seed_345)]
    tes_345_138=tline_tes_345['v'].drop_duplicates().values
    critical_tes_345_138=tes_138_data[tes_138_data['NODE_ID'].isin(tes_345_138)]
    
    seed_345e_data=selected_seed_data.loc[selected_seed_data['VOLTAGE'] == 345]
    seed_345e=seed_345e_data['NODE_ID'].values
    tline_tes_e345=edge_tline_tes[edge_tline_tes['u'].isin(seed_345e)]
    tes_e345_138=tline_tes_345['v'].drop_duplicates().values
    critical_tes_e345_138=tes_138_data[tes_138_data['NODE_ID'].isin(tes_e345_138)]
    
    print('tl_v>=345,tes_v>=138,tl_v>=345+tes_v>=138,tl_v==345+tes_v>=138')
    print(len(seed_345),len(tes_138),len(critical_tes_345_138),len(critical_tes_e345_138))
    #num_critical_tes=len(critical_tes)
    #print(critical_tes)
    
    critical_tes_data=trans_es[trans_es['NODE_ID'].isin(critical_tes)]
    print('total connected transmission substations:',critical_tes_data.shape[0])
    critical_tes_data.to_csv(outdir+reg+'_critical_tes.csv',index=False)

def evaluate_connected_es(outdir,transmissions,es,edge_tline_es,seeds,k):
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]
    
    tline_es=edge_tline_es[edge_tline_es['u'].isin(seeds)]
    
    critical_es=tline_es['v'].drop_duplicates().values
    es_data=es[es['NODE_ID'].isin(critical_es)]
    es_138_data=es_data.loc[es_data['MAX_KV']>=138]
    es_138=es_138_data['NODE_ID'].values
    
    seed_345_data=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    seed_345=seed_345_data['NODE_ID'].values
    tline_es_345=edge_tline_es[edge_tline_es['u'].isin(seed_345)]
    es_345_138=tline_es_345['v'].drop_duplicates().values
    critical_es_345_138=es_138_data[es_138_data['NODE_ID'].isin(es_345_138)]
    
    seed_345e_data=selected_seed_data.loc[selected_seed_data['VOLTAGE'] == 345]
    seed_345e=seed_345e_data['NODE_ID'].values
    tline_es_e345=edge_tline_es[edge_tline_es['u'].isin(seed_345e)]
    es_e345_138=tline_es_e345['v'].drop_duplicates().values
    critical_es_e345_138=es_138_data[es_138_data['NODE_ID'].isin(es_e345_138)]
    
    print('tl_v>=345,es_v>=138,tl_v>=345+es_v>=138,tl_v==345+es_v>=138')
    print(len(seed_345),len(es_138),len(critical_es_345_138),len(critical_es_e345_138))
    #num_critical_tes=len(critical_tes)
    #print(critical_tes)
    
    critical_es_data=es[es['NODE_ID'].isin(critical_es)]
    print('total connected distribution substations:',critical_es_data.shape[0])
    critical_es_data.to_csv(outdir+'critical_es.csv',index=False)
     
    
def evaluate_seeds_1_hop(transmissions,substations,trans_sub,seeds,
                         k,critical_fac,reg,is_voltage_rule,outdir=None):
    
    infile='data/v9/'
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
        
def check_network_robustness(filedir,outdir,outfile,edgefile,K,p,m,critical_fac,states,choice):
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                       delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    robust_file=open(outdir+outfile+'.csv','w')
    robust_file.writelines('reg,spread,k,lookup,v>=345,near_mil,v>=345+near-mil,near_hos,v>=345+near-hos\n')
    for reg in states:
        #transmissions=pd.read_csv(filedir+reg+'_Transmission_Lines.csv',\
         #                     delimiter=',',index_col=False,low_memory=False)
        print('State '+reg)
        start_time = time.perf_counter()
        nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
        nodes_list=csv.reader(nodes_file)
        nodes_map={int(row[1]):row[0] for row in nodes_list}
        nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
        df = pd.read_csv(filedir+reg+'_'+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
        df['act_prob']=df.kij*(1-df.pj)
        final_spread=-1
        for k in K:
            if choice==1: #full network not IM
                seed_list=get_original_tl_ids(nodes_map)
                k=len(seed_list)
            elif choice==2: #random model: pick k random nodes
                seed_list=get_random_tl_ids(nodes_map,k)
                lookup=len(seed_list)
            else:
                seed_list,seed_idx,final_spread,lookup = \
                    degreeDiscountIC(filedir,df,nodes_transmission,nodes_map,k,p,m)
                print('seeds',k,'spread',final_spread,'lookup',lookup)
        
            
            #evaluate_connected_tes(outdir,transmissions,trans_es,seed_list,k,reg)
            #evaluate_connected_es(outdir,transmissions,substations,trans_sub,seed_list,k)
        
            true_seed_c1,rulesc1,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list,k,critical_fac[0],reg,True,outdir=outdir)
            
            true_seed_c2,rulesc2,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list,k,critical_fac[1],reg,False)
            
            total_critical_seed=true_seed_c1+true_seed_c2
            #total_critical_seed1=set(total_critical_seed)
            total_rule=len(total_critical_seed)
            rulesc2['total_rule']=total_rule   
            print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
            print('total nodes:',len(list(nodes_map.keys())))
            print(rulesc1)
            print(rulesc2)
            #print('total seeds connected with all facilities:',total_rule)
            #'''    
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
            #'''
            '''
            if true_seed_c1.shape[0]>0:
                true_seed_c1.to_csv(outdir+reg+'_seed_near_mil.csv',index=False)
            if true_seed_c2.shape[0]>0:
                true_seed_c2.to_csv(outdir+reg+'_seed_near_hos.csv',index=False)
            
            
            with open(outdir+outfile+'_true_seed_c1_graph_'+reg+'.txt','w') as res:
                res.writelines(["%s\n" % item  for item in true_seed_c1])
            
            with open(outdir+outfile+'_true_seed_c2_graph_'+reg+'.txt','w') as res:
                res.writelines(["%s\n" % item  for item in true_seed_c2])
            '''   
    
if __name__=='__main__':
    filedir='data/G_for_robustness/' #sys.argv[1]
    outdir='output/G_vary_k/'#sys.argv[2]
    outfile='G_vary_k_random' #sys.argv[3] 
    #read_whole_graph(filedir,consumer1)
    m=10#sys.argv[4]
    K=[50,100,200,300,500]#sys.argv[5]
    #alpha: hyperparam choose stress increase on a node when its co-parent fails
    #alpha=0.2 #sys.argv[5]  
    
    p=0.1 #sys.argv[7]
    edgefile='all_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases','Hospitals']
    
    #state=['TX','GA','AL','VA','OH','TN','CA','FL','NY','PA','NM']
    #region=['EIC','WECC','NPCC','TX']
    region=['EIC','TX','national']
    choice=2 #1: full-network, 2:random 3: ic model
    #region=['national']
    check_network_robustness(filedir,outdir,outfile,edgefile,K,p,m,critical_fac,region,choice)
    