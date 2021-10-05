#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 08:25:50 2020

@author: anikat
"""
import networkx as nx
import re
import csv
import numpy as np
import pandas as pd
import random
import os

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def _get_translines_state(trans_id_list,nodes_map):
    '''
    arrayt=[]
    for tname in trans_id_list:
        try:
            tid=nodes_map[tname]
            arrayt.append(tid)
        except KeyError:
            print(tname+' not found:')
    '''
    return list(nodes_map.values())

def get_coin_prob_for_ic(filedir,edgefile):
    filename=filedir+'preprocessed_data/'+edgefile+'.txt'
    edge_data=pd.read_csv(filename,header=None,names=['u','v','kij','pj'])
    num_edges=edge_data.shape[0]
    samples=np.random.uniform(0,1,num_edges)
    edge_data['p']=samples
    
    edge_data.to_csv(filedir+'preprocessed_data/'+edgefile+'_p.txt',index=False)
    
def is_similar_type(n1,n2):
    return int(n1.split(':')[0]==n2.split(':')[0])

def get_kij(node,cur_par,G,nodes_map):
    sim=0
    for e in G.in_edges(node):
        sim+=is_similar_type(nodes_map[e[0]],nodes_map[cur_par])
    
    return 1/sim

def get_neighborhood_pj(node,par,G,nodes_map):
    #pj=1/sum(all the neighbors of same type)
    sim=1
    for u in G.in_edges(node): #u[0] is nodes parent
        for v in G.out_edges(u[0]):
            if v[1]!=node: #removing comparing node itself
                sim+=is_similar_type(nodes_map[node],nodes_map[v[1]])
            
    return 1/sim

def get_domain_constrain_pj(node,par,trans_volt_map,G,nodes_map,maxVolt=1000):
    b=get_neighborhood_pj(node,par,G,nodes_map) 
    node_nm=nodes_map[node]
    par_nm=nodes_map[par]
    no_domain=np.random.uniform(0,b,1)[0]
    if par_nm.split(':')[0]=='Transmission_Lines':
    #if node_nm.split(':')[0]=='Electric_Substations' or node_nm.split(':')[0]=='Transmission_Electric_Substations':
        if  trans_volt_map[par_nm]>0:
            return b, no_domain,np.abs((trans_volt_map[par_nm]-maxVolt))/maxVolt
        else:
            return b, no_domain,1 #removing considering unknown voltage from the nework  
    return b,no_domain,no_domain

    
def read_graph(filedir,edgedata,nodes_map,node_types):
    
    selected_edge=[False]*edgedata.shape[0]
    nodes_subgraph={}
    ttl_types={}
    for tp in node_types:
        ttl_types[tp]=0
        
    for ix,row in edgedata.iterrows():
        n1,n2=int(row['u']),int(row['v'])
        t1,t2=nodes_map[n1].split(':')[0],nodes_map[n2].split(':')[0]
        if t1 in node_types and t2 in node_types:
            selected_edge[ix]=True
            if n1 not in nodes_subgraph.keys():
                nodes_subgraph[n1]=nodes_map[n1]
                ttl_types[t1]+=1
            if n2 not in nodes_subgraph.keys():
                nodes_subgraph[n2]=nodes_map[n2]
                ttl_types[t2]+=1
    
    total_nodes=len(list(nodes_subgraph.keys()))
    
    for tp in node_types:
        print("# "+tp+": "+str(ttl_types[tp]))
    print("The # of all nodes: %d" %total_nodes)
    
    return nodes_subgraph,selected_edge

def write_node_edge_file(outdir,edgeix_data,region,selected_edge,nodes_subgraph):
    outfile=outdir+'all_edges_'+region+'.txt'
    f_edges=open(outfile,'w') 
    f_nodes=open(outdir+'all_nodes_index_'+region+'.txt','w') 
    num_edges=0
    num_nodes=len(list(nodes_subgraph.keys()))
    for ix,row in edgeix_data.iterrows():
        ei=int(row['u'])
        ej=int(row['v'])
        if selected_edge[ix] and ei!=ej:
            f_edges.write(str(ei)+','+str(ej)+'\n')
            num_edges+=1
    
    for node in nodes_subgraph.keys():
        f_nodes.write(nodes_subgraph[node]+','+str(node)+'\n')
    
    f_edges.close()
    f_nodes.close()
    return num_nodes, num_edges
    
def create_region_subgraph(filedir,regdir,edgefile,node_types,outdir,region):
    
    infile='../data/naerm/edge-files/'
    #trans_info_data=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False)
    trans_info_data=pd.read_csv(infile+'_Transmission_Substations.node',delimiter=',',index_col=False)
    trans_id=trans_info_data['NODE_ID'].values
    #trans_volt=trans_info_data['MAX_NOM_KV'].values
    trans_volt=trans_info_data['NOMKVMAX'].values
    trans_id_voltage=dict(zip(trans_id, trans_volt)) 
    
    edgeix_data=pd.read_csv(filedir+edgefile+'.txt',names=['u','v','kij','no_domain','pj','p'])
    
    nodes_file=open(filedir+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list} #int-string id
    
    
    nodes_str_int_map={nodes_map[key]:key for key in nodes_map.keys()} #int-string id
    nodes_tline=_get_translines_state(trans_id,nodes_str_int_map)
    
    ttl_trans=len(nodes_tline)
    print('ttl trans-line',ttl_trans)
    

    f_tline=open(outdir+'transmission_nodes_'+region+'.txt','w')
    for tline in nodes_tline:
        f_tline.write(str(tline)+'\n')
     
    nodes_subgraph,selected_edge=read_graph(filedir,edgeix_data,nodes_map,node_types) 
    
    ttl_nodes,ttl_edges=write_node_edge_file(outdir,edgeix_data,region,selected_edge,nodes_subgraph)
    reg_edge_data=pd.read_csv(outdir+'all_edges_'+region+'.txt',delimiter=',',names=['u','v'])
    G=nx.from_pandas_edgelist(reg_edge_data, 'u', 'v',create_using=nx.DiGraph())  
    
    print('total nodes:',ttl_nodes,'ttl edges:',ttl_edges)
    print('total transmission nodes:',ttl_trans)
    samples=np.random.uniform(0,1,ttl_edges)
    pj_map={}
    num_edges=0
    
    f_edges=open(outdir+region+'_'+edgefile+'.txt','w') 
    f_edges_check=open(outdir+region+'_'+edgefile+'_check.txt','w') 
    for e in G.edges():
        ei=nodes_subgraph[e[0]]
        ej=nodes_subgraph[e[1]]
        
        kij=get_kij(e[1],e[0],G,nodes_subgraph)
        b,no_domain,pj=get_domain_constrain_pj(e[1], e[0], trans_id_voltage, G,nodes_subgraph,maxVolt=999)
        #num_inc=1/kij
        if e[1] not in pj_map.keys():
            pj_map[e[1]]=pj
        f_edges.write(str(e[0])+','+str(e[1])+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
        f_edges_check.write(ei+','+ej+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
        num_edges+=1
    
    f_edges.close()
    f_edges_check.close()
    
    
if __name__=='__main__':
    filedir='../data/naerm/preprocessed_data/' #sys.argv[1]
    
    regdir='../data/naerm/edge-files/' #sys.argv[1]
    region=['national']
    edgefile='naerm_edges_rule_based_pj_p'
    node_types=['Transmission_Lines','Distribution_Substations','Transmission_Substations','Power_Plants']
    outdir='../data/naerm/national/'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    create_region_subgraph(filedir,regdir,edgefile,node_types,outdir,region[0])
    

    
    
    
