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

def draw_graph(G,consumer1,consumer2):
    import matplotlib.pyplot as plt
    val_map={'Power_Plants':'g','Transmission_Lines':'r',
             'Transmission_Electric_Substations':'r','Electric_Substations':'b',
             consumer1:'k',consumer2:'k'}
    
    values=[]
    for node in G.nodes():
        node_type=node.split(':')[0]
        values.append(val_map[node_type])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, cmap = plt.get_cmap('jet'),node_color=values)
    plt.show()
    
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

def get_coin_prob_for_ic(filedir,edgefile):
    filename=filedir+'preprocessed_data/'+edgefile+'.txt'
    edge_data=pd.read_csv(filename,header=None,names=['u','v','kij','pj'])
    num_edges=edge_data.shape[0]
    samples=np.random.uniform(0,1,num_edges)
    edge_data['p']=samples
    
    edge_data.to_csv(filedir+'preprocessed_data/'+edgefile+'_p.txt',index=False)
    
def generate_random_sample(sigma_vals,total_nodes,mu=0,sample_type='uniform'):
    '''
    Assumption mu=1, since ideal condition of protection system, 
    pij=0; CB has broken and failed to protect node 
    pij=1; CB is ideal condition and node cannot cascade. 
    Thus edge weight Pij=kij*(1-1)=0
    Since probability cannot be negative and will deviate from 0, 
    so assumed lognormal distribution
    expectation of lognormal E[X]= exp(mu+1/2 sigma^2)
    Variance of lognormal Var[x]= exp(2*mu+2*sigma^2)-exp(2*mu+sigma^2)
    '''
    sample_list={}
    if total_nodes>0:
        if sample_type=='uniform':
            for key in sigma_vals.keys():
                sigma=float(sigma_vals[key]/total_nodes)
                samples=np.random.uniform(mu,sigma,sigma_vals[key])
                sample_list[key]=samples
                #print('sigma',sigma,sigma_vals[key])
        elif sample_type=='lognormal':
            for key in sigma_vals.keys():
                sigma=float(sigma_vals[key]/total_nodes)
                samples=np.random.lognormal(mu,sigma,sigma_vals[key])
                sample_list[key]=samples
        elif sample_type=='normal':
            for key in sigma_vals.keys():
                sigma=float(sigma_vals[key]/total_nodes)
                samples=np.random.normal(mu,sigma,sigma_vals[key])
                sample_list[key]=samples
        #print('random-samples',sample_list)
    return sample_list

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

def get_kij(node,cur_par,G):
    sim=0
    for e in G.in_edges(node):
        sim+=is_similar_type(e[0],cur_par)
    
    return 1/sim

def get_neighborhood_pj(node,par,G):
    #pj=1/sum(all the neighbors of same type)
    sim=1
    for u in G.in_edges(node): #u[0] is nodes parent
        for v in G.out_edges(u[0]):
            if v[1]!=node: #removing comparing node itself
                sim+=is_similar_type(node,v[1])
         
    
    return 1/sim

def get_domain_constrain_pj(node,par,trans_volt_map,G,maxVolt=1000,add_domain=True):
    b=get_neighborhood_pj(node,par,G) 
    if add_domain:
        if par.split(':')[0]=='Transmission_Lines':
            if node.split(':')[0]=='Electric_Substations'\
                or node.split(':')[0]=='Transmission_Electric_Substations':
                if trans_volt_map[par]>0:
                    uj=np.abs((trans_volt_map[par]-maxVolt))/maxVolt
                    if uj>1:
                        uj=0
                    return b, uj
                else:
                    return b, 1 #removing considering unknown voltage from the nework  
    
    return b,np.random.uniform(0,b,1)[0]
    
def read_graph(filedir):
    edgelist = [f for f in glob.glob(filedir+"*.edge")]
    fall=open(filedir+"whole_network_edges_no_consumer.txt",'w') #write all edges into one graph
    data=''
    num_files=0
    for filename in edgelist:
        f1= open(filename,'r')
        #print(filename)
        curdata=f1.read()
        if num_files>0:
            data+='\n'
        data+=curdata
        num_files+=1
    
    fall.write(data) #write whole edge into a file
    fall.close()
    
    G=nx.read_edgelist(filedir+"whole_network_edges_no_consumer.txt", delimiter=',', create_using=nx.DiGraph(),nodetype=str) 
    #draw_graph(G,consumer1,consumer2)
    nodes_transmission=fuzzyfinder('Transmission',G)
    
    nodes_pplant=fuzzyfinder('Power_Plants',G)
    nodes_substation=fuzzyfinder('Electric_Substations',G)
    nodes_tes=fuzzyfinder('Transmission_Electric_Substations',G)
    nodes_tline=fuzzyfinder('Transmission_Lines',G)
    arrayall=nodes_pplant+nodes_transmission+nodes_substation #power systems
    '''
    PtoG_nodes=[]
    nodes_consumer1=[]
    
    if consumer1 is not None:
        nodes_consumer1=fuzzyfinder(consumer1,G) #Natural_Gas_Processing_Plants
        PtoG_nodes=nodes_substation+nodes_consumer1 
        #node power system to consumer substations+consumer1+consumer2   
    
    nodes_consumer2=[]
    if consumer2 is not None:
        nodes_consumer2=fuzzyfinder(consumer2,G)
        PtoG_nodes+=nodes_consumer2
    '''  
    DGT=G.subgraph(nodes_transmission) #network: transmission lines
    
    DGP=G.subgraph(arrayall)
    
    #PtoG=G.subgraph(PtoG_nodes) 
    #contain network of substations+gas compressor (consumer1)+consumer2 
    print("The # of powerplant nodes: %d" % len(nodes_pplant))
    print("The # of substation nodes: %d" % len(nodes_substation))    
    print("The # of transmission line: %d" % len(nodes_tline))    
    print("The # of transmission substation: %d" % len(nodes_tes))    
    print("The # of all nodes: %d" % len(arrayall))
    print("All nodes:%d" % len(G.nodes()))
    
    
    print("The # of transmission edges: %d" % len(DGT.edges()))

    print("The # of all power system edges: %d" % len(DGP.edges()))
    

    #print("The # of new all nodes in PtoG: %d" % len(PtoG.nodes()))
    #print("The # of new all edges in PtoG: %d" % len(PtoG.edges()))
    
    #write all string to integer mapping for all nodes
    path=filedir+'/new_preprocessed_data/'
    if not os.path.exists(path): 
        os.makedirs(path)
    nodes_file=open(path+'all_nodes_index.txt','w') #arrays
    transmission_nodes=open(path+'transmission_nodes.txt','w') #arrays
    index=1
    for i in nodes_pplant:
        nodes_file.write(i+','+str(index)+'\n')
        index+=1
    for i in nodes_tes:
        nodes_file.write(i+','+str(index)+'\n')
        index+=1
    for i in nodes_tline:
        nodes_file.write(i+','+str(index)+'\n')
        transmission_nodes.write(str(index)+'\n')
        index+=1
    for i in nodes_substation:
        nodes_file.write(i+','+str(index)+'\n')
        index+=1
    '''
    for i in nodes_consumer1:
        nodes_file.write(i+','+str(index)+'\n')
        index+=1
    for i in nodes_consumer2:
        nodes_file.write(i+','+str(index)+'\n')
        index+=1
    '''
    nodes_file.close()
    transmission_nodes.close()
    infile= open(path+'all_nodes_index.txt', mode='r')
    data=csv.reader(infile)
    #nodes_map={}
    #for rows in data:
     #   nodes_map[rows[0]]=int(rows[1])
    #print(nodes_map)
    nodes_map = {rows[0]:int(rows[1]) for rows in data}
    
    return G, nodes_map, nodes_pplant, nodes_substation,\
        nodes_transmission,nodes_tes,nodes_tline

def uncertainty_with_domain(filedir,edgefile):
    #calculate uncertainty for each node
    ###
    G, nodes_map, nodes_pplant, nodes_substation,\
        nodes_transmission,nodes_tes,nodes_tline=read_graph(filedir)
        
    infile='data/v9/'
    trans_info_data=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False)
    trans_id=trans_info_data['NODE_ID'].values
    trans_volt=trans_info_data['VOLTAGE'].values
    trans_id_voltage=dict(zip(trans_id, trans_volt)) 
    #selected_seed_data[['NODE_ID','VOLTAGE']]    
    '''
    1. computing the edge weight eij, Kij= 1/#incoming edges of node j of type 1 (i=type1) 
    2. writing edge files for each type of node with their integer id and edge-weight:node1,node2,weight
    '''
    
    f_edges=open(filedir+'/new_preprocessed_data/'+edgefile+'.txt','w') 
    f_edges_check=open(filedir+'/new_preprocessed_data/'+edgefile+\
                       '_check.txt','w') 
    f_edge_type=open(filedir+'/new_preprocessed_data/edge_type_kij_pj.csv','w') 
    #f_trans_sub.write('u,v,KV,ttl_KV,in_deg,kij\n')
    f_edge_type.write('u,v,in_deg,kij,b,pj\n')
    pj_map={}
    num_edges=0
    total_edges=len(list(G.edges()))
    samples=np.random.uniform(0,1,total_edges)
    for e in G.edges():
        ei=int(nodes_map[e[0]])
        ej=int(nodes_map[e[1]])
        #print(e)
        ti=e[0].split(':')[0]
        tj=e[1].split(':')[0]
        kij=get_kij(e[1],e[0],G)
        b,pj=get_domain_constrain_pj(e[1], e[0], trans_id_voltage, G)
        num_inc=1/kij
        if ej not in pj_map.keys():
            pj_map[ej]=pj
        f_edges.write(str(ei)+','+str(ej)+','+str(kij)+','+str(pj_map[ej])+','+str(samples[num_edges])+'\n')
        f_edges_check.write(e[0]+','+e[1]+','+str(kij)+','+str(pj_map[ej])+','+str(samples[num_edges])+'\n')
        f_edge_type.write(ti+','+tj+','+str(num_inc)+','+str(kij)+','\
                          +str(b)+','+str(pj_map[ej])+'\n')
        num_edges+=1
    f_edges.close()
    f_edges_check.close()
    f_edge_type.close()
    #print('uncertainty sample error in ',tj)

def uncertainty_with_network(filedir,edgefile,consumer1=None,consumer2=None):
    G, nodes_map, nodes_pplant, nodes_substation,\
        nodes_transmission,nodes_tes,nodes_tl\
            =read_graph(filedir,consumer1,consumer2)
    #calculate uncertainty for each node
    ###
    sigma_pplant,num_sample_pp=count_in_degree_dist(nodes_pplant,G,'pplant')
    sigma_substation,num_sample_ss=count_in_degree_dist(nodes_substation,G,'ES')
    sigma_tes,num_sample_tes=count_in_degree_dist(nodes_tes,G,'TES')
    sigma_tl,num_sample_tl=count_in_degree_dist(nodes_tl,G,'Tline')
    
    #sigma_consumer1,num_sample_c1=count_in_degree_dist(nodes_consumer1,G,'consumer1')
    #sigma_consumer2,num_sample_c2=count_in_degree_dist(nodes_consumer2,G,'consimer2')
    
    #get the random samples for uncertainty/protection probability
    pplant_pj=generate_random_sample(sigma_pplant,len(nodes_pplant))
    substation_pj=generate_random_sample(sigma_substation,len(nodes_substation))
    tes_pj=generate_random_sample(sigma_tes,len(nodes_tes))
    tl_pj=generate_random_sample(sigma_tl,len(nodes_tl))
    #consumer1_pj=generate_random_sample(sigma_consumer1,len(nodes_consumer1))
    #consumer2_pj=generate_random_sample(sigma_consumer2,len(nodes_consumer2))
    
    '''
    1. computing the edge weight eij, Kij= 1/#incoming edges of node j of type 1 (i=type1) 
    2. writing edge files for each type of node with their integer id and edge-weight:node1,node2,weight
    '''
    
    f_edges=open(filedir+'/preprocessed_data/'+edgefile+'.txt','w') 
    f_edges_check=open(filedir+'/preprocessed_data/'+edgefile+'_check.txt','w') 
    
    pj_map={} #dictionary to store uncertainty samples for each node
    for e in G.edges():
        ei=int(nodes_map[e[0]])
        ej=int(nodes_map[e[1]])
        #print(e)
        ti=e[0].split(':')[0]
        tj=e[1].split(':')[0]
        #if ti=='Transmission_Electric_Substations':
        #    ti='Transmission_Lines'
        
        #if tj=='Transmission_Electric_Substations':
        #   tj='Transmission_Lines'
        num_incj=G.in_degree(e[1])
        num_incij=0 #calculate type i indegrees of j
        for u,v in G.in_edges(e[1]):
            tu=u.split(':')[0]
            #if tu=='Transmission_Electric_Substations':
             #   tu='Transmission_Lines'
            if tu==ti:
                num_incij+=1
        
        kij=1/num_incij
        if ej not in pj_map.keys():
            if tj=='Power_plants':
                cur_idx=num_sample_pp[num_incj]
                uj=pplant_pj[num_incj][cur_idx]
                num_sample_pp[num_incj]+=1
            elif tj=='Electric_Substations':
                cur_idx=num_sample_ss[num_incj]
                uj=substation_pj[num_incj][cur_idx]
                num_sample_ss[num_incj]+=1
            elif tj=='Transmission_Lines':
                cur_idx=num_sample_tl[num_incj]
                uj=tl_pj[num_incj][cur_idx]
                num_sample_tl[num_incj]+=1
            elif tj=='Transmission_Electric_Substations':
                cur_idx=num_sample_tes[num_incj]
                uj=tes_pj[num_incj][cur_idx]
                num_sample_tes[num_incj]+=1
            '''
            elif tj==consumer1 and consumer1 is not None:
                cur_idx=num_sample_c1[num_incj]
                uj=consumer1_pj[num_incj][cur_idx]
                num_sample_c1[num_incj]+=1
            elif tj==consumer2 and consumer2 is not None:
                cur_idx=num_sample_c2[num_incj]
                uj=consumer2_pj[num_incj][cur_idx]
                num_sample_c2[num_incj]+=1
            '''
            pj_map[ej]=uj
        
        pj=pj_map[ej]
        #print(ei,ej,kij,pj)
        f_edges.write(str(ei)+','+str(ej)+','+str(kij)+','+str(pj)+'\n')
        f_edges_check.write(e[0]+','+e[1]+','+str(kij)+','+str(pj)+'\n')
    f_edges.close()
    f_edges_check.close()
        #print('uncertainty sample error in ',tj)

    
    
if __name__=='__main__':
    filedir='data/'#'data/toy/' #sys.argv[1]
    #consumer1='Natural_Gas_Compressor_Stations'#sys.argv[2]
    #consumer2=sys.argv[3]
    #edge_file='all_edges_random_pj'
    #uncertainty_with_network(filedir,edge_file)
    #edge_file='all_edges_no_rule_pj_p'
    edge_file='all_edges_rule_based_pj_p'
    uncertainty_with_domain(filedir,edge_file)
    #get_coin_prob_for_ic(filedir,edge_file)

    
    
    