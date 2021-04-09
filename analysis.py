#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 05:18:53 2021

@author: anikat
"""
import numpy as np
import pandas as pd
import os

def _get_translines_state(filename):
    arrayt=[]
    data=pd.read_csv(filename)
    tid_list=data['id'].drop_duplicates().values
    
    for tid in tid_list:
        tname='Transmission_Lines:'+str(tid)
        arrayt.append(tname)
        #print(tname)
    return arrayt

def evaluate_seeds_1_hop(transmissions,substations,trans_sub,seeds,
                         critical_fac,reg,is_voltage_rule,outdir=None):
    
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
        rules['v>=230']=vol_match2
        
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
    true_critical_data2=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 230]
    true_critical_data3=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 138]
    
    rules['v>=345+'+key]=len(true_critical_data1)
    rules['v>=230+'+key]=len(true_critical_data2)
    rules['v>=138+'+key]=len(true_critical_data3)
    
    return true_critical_data,rules,num_seed_to_sub


infile1='data/v9/'
infile2='data/G_for_robustness/'
reg=''
critical_fac=['Military_Bases','Hospitals']
k=50
outdir='data_analysis/'#sys.argv[2]

if not os.path.exists(outdir): 
        os.makedirs(outdir)

transmissions=pd.read_csv(infile1+reg+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
'''
trlist=[]
for ix,row in transmissions.iterrows():
      trlist.append('Transmission_Lines:'+str(row['id']))  

transmissions['NODE_ID']=trlist
'''
substations=pd.read_csv(infile1+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    
trans_sub=pd.read_csv(infile1+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)

#seed_list=_get_translines_state(infile2+reg+'_Transmission_Lines.csv')
seed_list=transmissions['NODE_ID'].drop_duplicates().values
print('total transmissions:',len(seed_list))
true_seed_c1,rulesc1,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list,critical_fac[0],reg,True)

true_seed_c2,rulesc2,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list,critical_fac[1],reg,False)
print(reg)
print(rulesc1)
print(rulesc2)
'''
if true_seed_c1.shape[0]>0:
    true_seed_c1.to_csv(outdir+reg+'_seed_near_mil.csv',index=False)

if true_seed_c2.shape[0]>0:
    true_seed_c2.to_csv(outdir+reg+'_seed_near_hos.csv',index=False)
'''      
