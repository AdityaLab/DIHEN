#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:58:31 2021

@author: anikat
"""
import pandas as pd
import numpy as np

def originial_list(nodes_map,node_type):
    orig_list=[]
    for key in nodes_map.keys():
        if nodes_map[key].split(':')[0]==node_type:
            orig_list.append(nodes_map[key])
    
    return orig_list

def evaluate_R7(trans_sub,dis_sub,seed_list,k,nodes_map):     
    
    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    dihen_dis=selected_dis_sub[selected_dis_sub['LOADMW']>=300.0]['NODE_ID'].values
    
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)]
    dihen_trans=selected_trans_sub[selected_trans_sub['LOADMW']>=300.0]['NODE_ID'].values
    original=127
    '''
    orig_seed_trans=originial_list(nodes_map,'Transmission_Substations')
    orig_seed_dis=originial_list(nodes_map,'Distribution_Substations')
    
    selected_orig_dis=dis_sub[dis_sub['NODE_ID'].isin(orig_seed_dis)]
    orig_dis=selected_orig_dis[selected_orig_dis['LOADMW']>=300.0]['NODE_ID'].values
    
    selected_orig_trans=trans_sub[trans_sub['NODE_ID'].isin(orig_seed_trans)]
    orig_trans=selected_orig_trans[selected_orig_trans['LOADMW']>=300.0]['NODE_ID'].values
    
    original_list=list(orig_dis)+list(orig_trans)
    original=len(original_list)
    print('original R7:',original)
    '''
    
    dihen_seed=list(dihen_dis)+list(dihen_trans)
    
    return original,len(dihen_seed)

def evaluate_R8(trans_sub,dis_sub,trans_to_dis,seed_list,nodes_map,k,critical_fac):
    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    
    selected_dis_sub=selected_dis_sub['NODE_ID'].values
    
    infile='../data/naerm/edge-files/'
    edges=pd.read_csv(infile+'_Distribution_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    dihen_seed=edges[edges['u'].isin(selected_dis_sub)]
    dihen_seed=dihen_seed['u'].drop_duplicates().values
    original=954
    '''
    orig_seed_list=originial_list(nodes_map,'Distribution_Substations')
    original_l=edges[edges['u'].isin(orig_seed_list)]
    original_l= original_l['u'].drop_duplicates().values
    original=len(original_l)
    print('original R8:',original)
    '''
    return original,len(dihen_seed)

def evaluate_R1(trans_sub,trans_line,seed_list,nodes_map,k):
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)] 
    dihen_trans_sub=selected_trans_sub[selected_trans_sub['NOMKVMAX']>=345.0]['NODE_ID'].values
    
    selected_trans_line=trans_line[trans_line['NODE_ID'].isin(seed_list)]
    dihen_trans_line=selected_trans_line[selected_trans_line['MAX_NOM_KV']>=345.0]['NODE_ID'].values
    
    dihen_seed=list(dihen_trans_sub)+list(dihen_trans_line)
    original=874
    '''
    orig_trans_sub=originial_list(nodes_map,'Transmission_Substations')
    orig_trans_line=originial_list(nodes_map,'Transmission_Lines')
    
    selected_orig_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(orig_trans_sub)]
    orig_list=selected_orig_trans_sub[selected_orig_trans_sub['NOMKVMAX']>=345.0]['NODE_ID'].values
    original=len(orig_list)
    print('original R1:',original)
    '''
    
    return original,len(dihen_seed)

def evaluate_R2R3(dis_sub,trans_sub,seed_list,k,critical_fac):
    infile='../data/naerm/edge-files/'

    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)]
    
    dis_critical_edge=pd.read_csv(infile+'_Distribution_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    trans_dis_edge= pd.read_csv(infile+'_Transmission_Substations-_Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    dis_seeds=selected_dis_sub['NODE_ID'].values
    trans_seeds=selected_trans_sub['NODE_ID'].values
    
    temp=trans_dis_edge[trans_dis_edge['u'].isin(trans_seeds)]
    dis_seeds2=temp['v'].drop_duplicates().values
    dis_seeds=list(dis_seeds)
    dis_seeds2=list(dis_seeds2)
    for s in dis_seeds2:
        if s not in dis_seeds:
            dis_seeds.append(s)
    
    
    selected_dis_military=dis_critical_edge[dis_critical_edge['u'].isin(dis_seeds)]
    selected_trans_for_dis=trans_dis_edge[trans_dis_edge['v'].isin(dis_seeds)]
    
    #trans_seeds2=selected_dis_for_trans['v'].drop_duplicates().values
    #selected_dis_military2=dis_critical_edge[dis_critical_edge['u'].isin(trans_seeds2)]
    
    selected_dis=selected_dis_military['u'].drop_duplicates().values
    selected_trans=selected_trans_for_dis['u'].drop_duplicates().values
    #print('#seed-connected substations connected to critical facility:',seeded_subs_military.shape)
    
    selected_r3=trans_sub[trans_sub['NODE_ID'].isin(selected_trans)]
    selected_r3=selected_r3.loc[selected_r3['NOMKVMAX']>=345.0]
    
    original_r2=3157
    original_r3=100#74
    
    '''
    line_dis_edge= pd.read_csv(infile+'_Transmission_Lines-Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    trans_line= pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False,low_memory=False)
    
    
    original_r2=dis_critical_edge['u'].drop_duplicates().values
    original_connected_trans=trans_dis_edge[trans_dis_edge['v'].isin(original_r2)]
    original_connected_line= line_dis_edge[line_dis_edge['v'].isin(original_r2)]
    original_trans=original_connected_trans['u'].drop_duplicates().values
    original_line=original_connected_line['u'].drop_duplicates().values
    original_r3_trans=trans_sub[trans_sub['NODE_ID'].isin(original_trans)]
    original_r3_trans=original_r3_trans.loc[original_r3_trans['NOMKVMAX']>=345.0]
    original_r3_line=trans_line[trans_line['NODE_ID'].isin(original_line)]
    original_r3_line=original_r3_line.loc[original_r3_line['MAX_NOM_KV']>=345.0]
    
    original_r2=len(original_r2)
    original_r3=len(original_r3_trans)+len(original_r3_line)
    
    print('original R2, R3')
    print(original_r2,original_r3)
    ''' 
    return len(selected_dis),len(selected_r3)
