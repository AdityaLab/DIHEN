#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:43:49 2021

@author: anikat1
"""
import os
import numpy as np
import networkx as nx
import time
import csv
import pandas as pd
#from independent_cascade import independent_cascade
from degreeDiscountIC import degreeDiscountIC
import ev_criticality_criteria as CR

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    
    return arrayt            
'''   
def get_act_prob(data,nodes_map,b1,check_rule):
    #check_rule==True, use b2 for edges transmission->substation
    #check_rule==False, use b2 for edges not transmission->substation
    act_prob=[]
    for ix,row in data.iterrows():
        #print(row['u'],row('v'))
        n1,n2=row['u'],row['v']
        u,v=nodes_map[n1],nodes_map[n2]
        tu,tv=u.split(':')[0],v.split(':')[0]
        if tu=='Load_Buses' or tu=='Non_Load_Buses' or tu=='Substations':
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
'''
def scalability_k_size(path,outdir,outfile,edgefile,reg,K,p,m,critical_fac):
    infile='../data/naerm/new-graph-EV/'
    
    load_bus=pd.read_csv(infile+'_Load_Buses.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    sub=pd.read_csv(infile+'_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    sub_to_load=pd.read_csv(infile+'_Substations-Load_Buses.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)

    time_file=open(outdir+outfile+'.csv','w')
    time_file.writelines('k,num_node,num_edge,spread,model_time,eval_time\n')
    
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(path+'all_nodes_index_1'+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_sub=_get_array(path+'sub_nodes_1'+'.txt')
    nodes_bus=_get_array(path+'bus_nodes_1'+'.txt')
    nodes_sub=nodes_sub+nodes_bus
    edgef='graph_1'+edgefile
    df = pd.read_csv(filedir+edgef+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    
    df['act_prob']=df.kij*(1-df.pj)
    num_nodes=len(list(nodes_map.keys()))
    num_edges=df.shape[0]
    for k in K:
        print('k:',k,num_nodes,df.shape[0])
        seed_list,seed_idx,final_spread,lookup = \
        degreeDiscountIC(path,df,nodes_sub,nodes_map,k,p,m,False)
        stop1=  (time.perf_counter() - start_time)   
        original_R7,R7 = CR.evaluate_R7(load_bus,sub,sub_to_load,seed_list,k,nodes_map)
            
        original_R8,R8=CR.evaluate_R8(load_bus,sub,sub_to_load,seed_list,nodes_map,k,critical_fac[0])
            
        original_R1,R1 = CR.evaluate_R1(load_bus,sub,seed_list,nodes_map,k)
        
        R2,R3=CR.evaluate_R2R3(sub,load_bus,seed_list,k,critical_fac[1])
        stop2=  (time.perf_counter() - start_time)   
        print("Running time:%s sec %s sec" % (stop1,stop2))
        print(len(seed_list),final_spread)
            
        print(R1,R2,R3,R7,R8)
            
        string1=str(k)+','+str(num_nodes)+','
        string2=str(num_edges)+','+str(final_spread)+','
        string3=str(stop1)+','+str(stop2)+'\n'
            
        time_file.writelines(string1+string2+string3)       
            
        #'''
        with open(outdir+'graph_1_seeds_'+str(k)+'.txt', 'w') as f:
            for idx in range(0,len(seed_list)):
                f.write("%s,%d\n" % (seed_list[idx],seed_idx[idx]))
        #'''
    time_file.close()
                       
def scalability_graph_size(path,outdir,outfile,edgefile,reg,k,p,m,critical_fac,cases):
    infile='../data/naerm/new-graph-EV/'
    
    load_bus=pd.read_csv(infile+'_Load_Buses.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    sub=pd.read_csv(infile+'_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    sub_to_load=pd.read_csv(infile+'_Substations-Load_Buses.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)

    time_file=open(outdir+outfile+'.csv','w')
    time_file.writelines('graph_id,k,num_node,num_edge,spread,model_time,eval_time\n')
    
    print('Begin:')
    for gid in cases:
        start_time = time.perf_counter()
        nodes_file=open(path+'all_nodes_index_'+str(gid)+'.txt','r')
        nodes_list=csv.reader(nodes_file)
        nodes_map={int(row[1]):row[0] for row in nodes_list}
        nodes_sub=_get_array(path+'sub_nodes_'+str(gid)+'.txt')
        nodes_bus=_get_array(path+'bus_nodes_'+str(gid)+'.txt')
        nodes_sub=nodes_sub+nodes_bus
        edgef='graph_'+str(gid)+edgefile
        df = pd.read_csv(filedir+edgef+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    
        df['act_prob']=df.kij*(1-df.pj)
        num_nodes=len(list(nodes_map.keys()))
        num_edges=df.shape[0]
        print('graph:',gid,num_nodes,df.shape[0])
        seed_list,seed_idx,final_spread,lookup = \
        degreeDiscountIC(path,df,nodes_sub,nodes_map,k,p,m,False)
        stop1=  (time.perf_counter() - start_time)   
        original_R7,R7 = CR.evaluate_R7(load_bus,sub,sub_to_load,seed_list,k,nodes_map)
            
        original_R8,R8=CR.evaluate_R8(load_bus,sub,sub_to_load,seed_list,nodes_map,k,critical_fac[0])
            
        original_R1,R1 = CR.evaluate_R1(load_bus,sub,seed_list,nodes_map,k)
        
        R2,R3=CR.evaluate_R2R3(sub,load_bus,seed_list,k,critical_fac[1])
        original_R9,R9 = -1,-1
        stop2=  (time.perf_counter() - start_time)   
        print("Running time:%s sec %s sec" % (stop1,stop2))
        print(len(seed_list),final_spread)
            
        print(R1,R2,R3,R7,R8,R9)
            
        string1=str(gid)+','+str(len(seed_list))+','+str(num_nodes)+','
        string2=str(num_edges)+','+str(final_spread)+','
        string3=str(stop1)+','+str(stop2)+'\n'
            
        time_file.writelines(string1+string2+string3)
            
            
        #'''
        with open(outdir+'graph_'+str(gid)+'_seeds_'+str(k)+'.txt', 'w') as f:
            for idx in range(0,len(seed_list)):
                f.write("%s,%d\n" % (seed_list[idx],seed_idx[idx]))
        #'''
    time_file.close()
           
    
if __name__=='__main__':
    filedir='../data/naerm/new-graph-EV/scalability-graph/' #sys.argv[1]
    outdir='../output/naerm-EV/scalability/'#sys.argv[2]
   
    outfile1='scalability_G' #sys.argv[3] 
    outfile2='scalability_k'
    m=10
    k=50#sys.argv[4]
    K=[50,100,200,300,500]
    p=0.1 
    edgefile='_naerm_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Compressor_Stations','Military_Bases']
    region='national'
    graph_id=[1,2,3,4,5]
    scalability_graph_size(filedir,outdir,outfile1,edgefile,region,k,p,m,critical_fac,graph_id)
    scalability_k_size(filedir,outdir,outfile2,edgefile,region,K,p,m,critical_fac)
    
