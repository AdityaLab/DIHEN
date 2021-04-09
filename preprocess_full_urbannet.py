#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 08:25:50 2020

@author: anikat
"""
import networkx as nx
import re
import glob
import csv
import numpy as np
import pandas as pd
import os

'''   
def read_graph(filedir):
    edgelist = [f for f in glob.glob(filedir+"*.edge")]
    fall=open(filedir+"whole_urbannet_edges.txt",'w') #write all edges into one graph
    data=''
    num_files=0
    node_name={}
    num_layers=0
    for filename in edgelist:
        names=filename.split('/')
        node1,node2=names[-1].split('-')
        node1=node1[1:]
        node2=node2.replace('.edge','')
        if os.path.getsize(filename) > 0:
            if node1 not in node_name.keys():
                num_layers+=1
                node_name[node1]=num_layers
            if node2 not in node_name.keys():
                num_layers+=1
                node_name[node2]=num_layers
            f1= open(filename,'r')
            #print(filename)
            curdata=f1.read()
            if num_files>0:
                data+='\n'
            data+=curdata
            num_files+=1
    #print(num_files)
    fall.write(data) #write whole edge into a file
    fall.close()
    
    G=nx.read_edgelist(filedir+"whole_urbannet_edges.txt", delimiter=',', create_using=nx.DiGraph(),nodetype=str) 
    #draw_graph(G,consumer1,consumer2)
    nodes_transmission=fuzzyfinder('Transmission',G)
    
    nodes_pplant=fuzzyfinder('Power_Plants',G)
    nodes_substation=fuzzyfinder('Electric_Substations',G)
    nodes_tes=fuzzyfinder('Transmission_Electric_Substations',G)
    nodes_tline=fuzzyfinder('Transmission_Lines',G)
    
    #arraypp=nodes_pplant+nodes_transmission+nodes_substation #power systems
    
    #arrayall=[]
    nodes_list={}
    for name in node_name.keys():
        nodes=fuzzyfinder(name,G)
        nodes_list[name]=nodes
        #arrayall=arrayall+nodes
        
    DGT=G.subgraph(nodes_transmission) #network: transmission lines
    
    #DGP=G.subgraph(arrayall)
    
    #PtoG=G.subgraph(PtoG_nodes) 
    #contain network of substations+gas compressor (consumer1)+consumer2 
    print("The # of powerplant nodes: %d" % len(nodes_pplant))
    print("The # of substation nodes: %d" % len(nodes_substation))    
    print("The # of transmission line: %d" % len(nodes_tline))    
    print("The # of transmission substation: %d" % len(nodes_tes))    
    print("The # of all nodes: %d" % (len(G.nodes)))
    
    
    print("The # of transmission edges: %d" % len(DGT.edges()))

    print("The # of all urbannet edges: %d" % len(G.edges()))
    

    #print("The # of new all nodes in PtoG: %d" % len(PtoG.nodes()))
    #print("The # of new all edges in PtoG: %d" % len(PtoG.edges()))
    
    #write all string to integer mapping for all nodes
    path=filedir+'/preprocessed_data/'
    if not os.path.exists(path): 
        os.makedirs(path)
    nodes_file=open(path+'all_nodes_index.txt','w') #arrays
    transmission_nodes=open(path+'transmission_nodes.txt','w') #arrays
    index=1
    for name in node_name.keys():
        nodes_urban=nodes_list[name]
        for i in nodes_urban:
            nodes_file.write(i+','+str(index)+'\n')
            if name=='Transmission_Lines':
                transmission_nodes.write(str(index)+'\n')
            index+=1
        
    nodes_file.close()
    transmission_nodes.close()
    infile= open(path+'all_nodes_index.txt', mode='r')
    data=csv.reader(infile)
    nodes_map = {rows[0]:int(rows[1]) for rows in data}
    
    return G, nodes_map,nodes_transmission,nodes_tes,nodes_tline
''' 

def fuzzyfinder(user_input, collection):
    suggestions = []
    pattern = '.*'.join(user_input) # Converts 'djm' to 'd.*j.*m'
    regex = re.compile(pattern)     # Compiles a regex.
    for item in collection:
        match = regex.search(item)  # Checks if the current item matches the regex.
        if match:
            if user_input=='Electric_Substations':
                if 'Transmission' not in item:
                    suggestions.append(item)
            else:
                suggestions.append(item)
    #print('without removing duplicates # nodes in '+user_input+' '+str(len(suggestions)))
    return list(set(suggestions))
    #return suggestions

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

def get_domain_constrain_pj(node,par,trans_volt_map,es_volt_map,G,nodes_map,maxVolt=1000):
    b=get_neighborhood_pj(node,par,G,nodes_map) 
    par_nm=nodes_map[par]
    node_nm=nodes_map[node]
    no_domain=np.random.uniform(0,b,1)[0]
    if par_nm.split(':')[0]=='Transmission_Lines':
        if node_nm.split(':')[0]=='Electric_Substations'\
            or node_nm.split(':')[0]=='Transmission_Electric_Substations':
            if trans_volt_map[par_nm]>0:
                uj=np.abs((trans_volt_map[par_nm]-maxVolt))/maxVolt
                if uj>1:
                    uj=0
                return b, no_domain, uj
            #return b,np.abs((trans_volt_map[par]-maxVolt))/maxVolt
            else:
                return b,no_domain,1 #removing considering unknown voltage from the nework  
    elif par_nm.split(':')[0]=='Electric_Substations':
        uj=np.abs((es_volt_map[par_nm]-maxVolt))/maxVolt
        if uj>1:
            uj=0
    return b,no_domain,no_domain

   
def read_matt_graph(filedir):
    path=filedir+'/preprocessed_data/'
    if not os.path.exists(path): 
        os.makedirs(path)
    nodes_list=open(filedir+'index_seq.txt','r')
    nodes_file=open(path+'all_nodes_index.txt','w')
    nodes_map={}
    num_trans=0
    transmission_nodes=open(path+'transmission_nodes.txt','w') #arrays
    G=nx.read_edgelist(filedir+"whole-graph.txt", delimiter=' ', create_using=nx.DiGraph(),nodetype=int) 
    for line in nodes_list:
        node_name,node_id=line.split(' ')
        node_id=int(node_id)
        if node_id not in nodes_map.keys():
            nodes_map[node_id]=node_name
            nodes_file.write(node_name+','+str(node_id)+'\n')
            if node_name.split(':')[0]=='Transmission_Lines':
                transmission_nodes.write(str(node_id)+'\n')
                num_trans+=1
    
    return G,nodes_map,num_trans
        
        
def uncertainty_with_domain(filedir,edgefile):
    #calculate uncertainty for each node
    ###
    #G, nodes_map,nodes_transmission,nodes_tes,nodes_tline =read_graph(filedir)
    G, nodes_map,total_trans=read_matt_graph(filedir)
    path=filedir+'/preprocessed_data/'
    infile='data/v9/'
    trans_info_data=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False,low_memory=False)
    trans_id=trans_info_data['NODE_ID'].values
    trans_volt=trans_info_data['VOLTAGE'].values
    trans_id_voltage=dict(zip(trans_id, trans_volt)) 
    
    es_info_data=pd.read_csv(infile+'_Electric_Substations.node',delimiter=',',index_col=False,low_memory=False)
    es_id=es_info_data['NODE_ID'].values
    es_volt=es_info_data['NOMINAL_KV'].values
    es_id_voltage=dict(zip(es_id, es_volt))
    #selected_seed_data[['NODE_ID','VOLTAGE']]    
    '''
    1. computing the edge weight eij, Kij= 1/#incoming edges of node j of type 1 (i=type1) 
    2. writing edge files for each type of node with their integer id and edge-weight:node1,node2,weight
    '''
    total_edges=len(list(G.edges()))
    total_nodes=len(list(nodes_map.keys()))
    
    print('Total nodes:',total_nodes)
    print('Total edges:',total_edges)
    print('Total transmission lines:',total_trans)
    samples=np.random.uniform(0,1,total_edges)
    pj_map={}
    f_edges=open(path+edgefile+'.txt','w') 
    f_edges_check=open(path+edgefile+'_check.txt','w') 
    f_edge_type=open(path+'edge_type_kij_pj_p.csv','w') 
    #f_trans_sub.write('u,v,KV,ttl_KV,in_deg,kij\n')
    f_edge_type.write('u,v,in_deg,kij,b,pj,p\n')
    num_edges=0
    for e in G.edges():
        #ei=int(nodes_map[e[0]])
        #ej=int(nodes_map[e[1]])
        ei=nodes_map[e[0]] #ei is node name
        ej=nodes_map[e[1]] #ej is node name 
        #print(e)
        ti=ei.split(':')[0]
        tj=ej.split(':')[0]
        kij=get_kij(e[1],e[0],G,nodes_map)
        b,no_domain,pj=get_domain_constrain_pj(e[1],e[0],trans_id_voltage,es_id_voltage,G,nodes_map)
        num_inc=1/kij
        if e[1] not in pj_map.keys():
            pj_map[e[1]]=pj
        f_edges.write(str(e[0])+','+str(e[1])+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
        f_edges_check.write(ei+','+ej+','+str(kij)+','+str(no_domain)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
        f_edge_type.write(ti+','+tj+','+str(num_inc)+','+str(kij)+','\
                          +str(b)+','+str(pj_map[e[1]])+','+str(samples[num_edges])+'\n')
    
        num_edges+=1
        
    f_edges.close()
    f_edges_check.close()
    f_edge_type.close()
    print('Finished processing network')

if __name__=='__main__':
    filedir='data/v9/urbannet2020-graph-v9/'#'data/toy/' #sys.argv[1]
    #edge_file='all_edges_random_pj'
    edge_file='urbannet_edges_rule_based_pj_p_v2' #added substation volts
    uncertainty_with_domain(filedir,edge_file)

    
    
    