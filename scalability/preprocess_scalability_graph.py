#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:27:26 2021

@author: anikat1
"""

import networkx as nx
import re
import glob
import csv
import numpy as np
import pandas as pd
import os

def _get_translines_state(trans_id_list,nodes_map):
    
    arrayt=[]
    for tname in trans_id_list:
        try:
            tid=nodes_map[tname]
            arrayt.append(tid)
        except KeyError:
            print(tname+' not found:')
    
    return arrayt

def count_in_degree_dist(nodes_list,graph,layer_type):
    '''
    calculate sigma for random sampling, 
    where sigma=# same type nodes that have same
    '''
    sigma_vals={}
    num_samples={}
    if len(nodes_list)>0:
        in_degrees=graph.in_degree(nodes_list)
        in_degrees=dict(in_degrees)
        #print(' in-deg:',in_degrees)
        degree_vals=sorted(set(in_degrees.values()))
        in_deg_dist = [list(in_degrees.values()).count(x) for x in degree_vals]
        in_deg_dist=np.array(in_deg_dist)
        #in_deg_dist/=len(nodes_list)
        sigma_vals = dict(zip(degree_vals, in_deg_dist))
        num_samples=dict(zip(degree_vals,[0]*len(degree_vals)))
        print(layer_type+str(' in-degree:'))
        print(sigma_vals)
    return sigma_vals,num_samples

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

def get_domain_constrain_pj(node,par,trans_volt_map,es_volt_map,G,nodes_map,maxVolt=765):
    b=get_neighborhood_pj(node,par,G,nodes_map) 
    par_nm=nodes_map[par]
    #node_nm=nodes_map[node]
    no_domain=np.random.uniform(0,b,1)[0]
    if par_nm.split(':')[0]=='Load_Buses' or par_nm.split(':')[0]=='Non_Load_Buses':
        #if node_nm.split(':')[0]=='Substations'\
         #   or node_nm.split(':')[0]=='Non_Load_Buses':
        if trans_volt_map[par_nm]>0:
            uj=np.abs((trans_volt_map[par_nm]-maxVolt))/maxVolt
            if uj>1:
                uj=0
            return b, no_domain, uj
        else:
            return b,no_domain,1 #removing considering unknown voltage from the nework  
    elif par_nm.split(':')[0]=='Substations':
        uj=np.abs((es_volt_map[par_nm]-maxVolt))/maxVolt
        if uj>1:
            uj=0
    return b,no_domain,no_domain

def read_graph(filedir,edgedata,nodes_map,node_types):
    print(node_types)
    selected_edge=[False]*edgedata.shape[0]
    nodes_subgraph={}
    #ttl_types={}
    #for tp in node_types:
    #    ttl_types[tp]=0
        
    for ix,row in edgedata.iterrows():
        n1,n2=int(row['u']),int(row['v'])
        t1,t2=nodes_map[n1].split(':')[0],nodes_map[n2].split(':')[0]
        if node_types[-1]=='All':
            selected_edge[ix]=True
            nodes_subgraph[n1]=nodes_map[n1]
            nodes_subgraph[n2]=nodes_map[n2]
        elif t1 in node_types and t2 in node_types:
            selected_edge[ix]=True
            if n1 not in nodes_subgraph.keys():
                nodes_subgraph[n1]=nodes_map[n1]
                #ttl_types[t1]+=1
            if n2 not in nodes_subgraph.keys():
                nodes_subgraph[n2]=nodes_map[n2]
                #ttl_types[t2]+=1
    
    total_nodes=len(list(nodes_subgraph.keys()))
    
    #for tp in node_types:
     #   print("# "+tp+": "+str(ttl_types[tp]))
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
    
def create_region_subgraph(filedir,edgefile,node_types,outdir,region):
    infile='../data/naerm/new-graph-EV/'
    trans1_info_data=pd.read_csv(infile+'_Load_Buses.node',delimiter=',',index_col=False,low_memory=False)
    trans2_info_data=pd.read_csv(infile+'_Non_Load_Buses.node',delimiter=',',index_col=False,low_memory=False)
    trans_id=list(trans1_info_data['NODE_ID'].values)+list(trans2_info_data['NODE_ID'].values)
    trans_volt=list(trans1_info_data['VOLT_KV'].values)+list(trans2_info_data['VOLT_KV'].values)
    trans_id_voltage=dict(zip(trans_id, trans_volt)) 

    es_info_data=pd.read_csv(infile+'_Substations.node',delimiter=',',index_col=False,low_memory=False)
    es_id=es_info_data['NODE_ID'].values
    es_volt=es_info_data['NOMINAL_KV'].values
    es_id_voltage=dict(zip(es_id, es_volt))
    
    edgeix_data=pd.read_csv(filedir+edgefile+'.txt',names=['u','v','kij','no_domain','pj','p'])
    
    nodes_file=open(filedir+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list} #int-string id
    
    
    nodes_str_int_map={nodes_map[key]:key for key in nodes_map.keys()} #int-string id
    nodes_tline=_get_translines_state(trans_id,nodes_str_int_map)
    nodes_sub=_get_translines_state(es_id,nodes_str_int_map)
    
    ttl_trans=len(nodes_tline)
    print('ttl trans-line',ttl_trans)
    graph_id=0
    for g in range(3,len(node_types),2):
        graph_id+=1
        f_tline=open(outdir+'bus_nodes_'+str(graph_id)+'.txt','w')
        for tline in nodes_tline:
            f_tline.write(str(tline)+'\n')
    
        f_tline=open(outdir+'sub_nodes_'+str(graph_id)+'.txt','w')
        for tline in nodes_sub:
            f_tline.write(str(tline)+'\n')
    
        nodes_subgraph,selected_edge=read_graph(filedir,edgeix_data,nodes_map,node_types[:g+1]) 
    
        ttl_nodes,ttl_edges=write_node_edge_file(outdir,edgeix_data,str(graph_id),selected_edge,nodes_subgraph)
        reg_edge_data=pd.read_csv(outdir+'all_edges_'+str(graph_id)+'.txt',delimiter=',',names=['u','v'])
        G=nx.from_pandas_edgelist(reg_edge_data, 'u', 'v',create_using=nx.DiGraph())  
    
        print('graph id:',str(graph_id),'total nodes:',ttl_nodes,'ttl edges:',ttl_edges)
        print('total transmission nodes:',ttl_trans)
        samples=np.random.uniform(0,1,ttl_edges)
        pj_map={}
        num_edges=0
    
        f_edges=open(outdir+'graph_'+str(graph_id)+'_'+edgefile+'.txt','w') 
        f_edges_check=open(outdir+'graph_'+str(graph_id)+'_'+edgefile+'_check.txt','w') 
        for e in G.edges():
            ei=nodes_subgraph[e[0]]
            ej=nodes_subgraph[e[1]]
        
        
            kij=get_kij(e[1],e[0],G,nodes_subgraph)
            b,no_domain,pj=get_domain_constrain_pj(e[1],e[0],trans_id_voltage,es_id_voltage,G,nodes_subgraph)
            #num_inc=1/kij
            if e[1] not in pj_map.keys():
                pj_map[e[1]]=pj
            f_edges.write(str(e[0])+','+str(e[1])+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
            f_edges_check.write(ei+','+ej+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
            num_edges+=1
    
        f_edges.close()
        f_edges_check.close()
    

if __name__=='__main__':
    filedir='../data/naerm/new-graph-EV/' #sys.argv[1]
    
    #regdir='../data/naerm/new-graph-EV/' #sys.argv[1]
    region=['national']
    edgefile='naerm_edges_rule_based_pj_p'
    node_types=['Load_Buses','Substations','Generators','Non_Load_Buses',
                'Military_Bases','Fire_Stations',
                'Compressor_Stations','Natural_Gas_Pipelines',
                'Processing_Plants','Wastewater_Treatment_Plants','All','All']
    outdir='../data/naerm/new-graph-EV/scalability-graph/'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    create_region_subgraph(filedir,edgefile,node_types,outdir,region[0])
    
    
    
