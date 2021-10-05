#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 2 13:13:02 2021

@author: anikat
This script is to get results vary size of k on domain-based national network
"""
#import sys
import os
import numpy as np
import networkx as nx
import time
import csv
import pandas as pd
#from independent_cascade import independent_cascade
from degreeDiscountIC import degreeDiscountIC
import naerm_critical as CR

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    
    return arrayt
    
def get_act_prob(data,nodes_map,b1,check_rule):
    #check_rule==True, use b2 for edges transmission->substation
    #check_rule==False, use b2 for edges not transmission->substation
    act_prob=[]
    for ix,row in data.iterrows():
        #print(row['u'],row('v'))
        n1,n2=row['u'],row['v']
        u,v=nodes_map[n1],nodes_map[n2]
        tu,tv=u.split(':')[0],v.split(':')[0]
        if tu=='Transmission_Lines' and tv=='Distribution_Substations'\
            or tv=='Transmission_Substations':
            if check_rule:
                pij=b1*(1-row['pj'])
            else:
                pij=(1-row['no_domain'])
                
        else:
            if check_rule:
                pij=b1*(1-row['pj'])
            else:
                pij=1-row['no_domain']
        act_prob.append(pij)
    
    return act_prob
            
def check_k_robustness(path,outdir,outfile,edgefile,reg,K,p,m,critical_fac):
    infile='../data/naerm/edge-files/'
    trans_line=pd.read_csv(infile+'_Transmission_Lines.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    dis_sub=pd.read_csv(infile+'_Distribution_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_to_dis=pd.read_csv(infile+'_Transmission_Substations-_Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    robust_file=open(outdir+outfile+'.csv','w')
    robust_file.writelines('k,spread,lookup,R1,R2,R3,R7,R8,R9\n')
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(path+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes_'+reg+'.txt')
    
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    
    df['act_prob']=df.kij*(1-df.pj)
    
    for k in K:
            print('k:',k)
            tmp_data=df.copy()
    
            seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,k,p,m,False)
            
            original_R7,R7 = CR.evaluate_R7(trans_sub,dis_sub,seed_list,k,nodes_map)
            
            original_R8,R8=CR.evaluate_R8(trans_sub,dis_sub,trans_to_dis,seed_list,nodes_map,k,critical_fac[0])
            
            original_R1, R1 = CR.evaluate_R1(trans_sub,trans_line,seed_list,nodes_map,k)
            
            R2,R3=CR.evaluate_R2R3(dis_sub,trans_sub,seed_list,k,critical_fac[1])
            original_R9,R9 = -1,-1
            print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
            print(len(seed_list),final_spread,lookup)
            
            print(R1,R2,R3,R7,R8,R9)
            
            string1=str(k)+','+str(final_spread)+','+str(lookup)+','
            string2=str(R1)+','+str(R2)+','+str(R3)+','
            string3=str(R7)+','+str(R8)+','+str(R9)+'\n'
            
            
            robust_file.writelines(string1+string2+string3)
            
            
            
    '''
    with open(outdir+'naerm_seeds_'+str(K[0])+'.txt', 'w') as f:
        for idx in range(0,len(seed_list)):
            f.write("%s,%d\n" % (seed_list[idx],seed_idx[idx]))
    '''
    robust_file.close()
           
    
if __name__=='__main__':
    filedir='../data/naerm/naerm_regional/' #sys.argv[1]
    outdir='../output/naerm/'#sys.argv[2]
   
    outfile='naerm_original_national' #sys.argv[3] 
    m=10
    K=[50,100,200,300,500]
    p=0.1 
    edgefile='national_naerm_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Compressor_Stations','Military_Bases']
    region='national'
    
    check_k_robustness(filedir,outdir,outfile,edgefile,region,K,p,m,critical_fac)
    
