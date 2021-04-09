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
import math
import preprocess_graph as prepg
import csv
import pandas as pd
from degreeDiscountIC import degreeDiscountIC
#import pickle

'''    
def degreeDiscountIC(indir,outdir,outfile,edgefile,k,p,m=10,writeFile=True):
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes.txt')
    df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj'])
    df['act_prob']=b1*df.kij*b2*(1-df.pj)
    G=nx.from_pandas_edgelist(df, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
    
    print('Total nodes:',len(G.nodes()))
    print('Total edges:',len(G.edges()))
    
    S0=[]
    S0_avg=[]
    d={} #out_degrees
    dd={} #degree_discount
    t={}
    Maxnodes=len(G.nodes())
    Max_trans_nodes=len(nodes_transmission)
    cur=[False]*Maxnodes
    all_nodes=list(G.nodes())
    for u in all_nodes:
        tmp=0
        #print('edges %d:%d'%(u,len(list(G.out_edges(u)))))
        for v in G.out_edges(u):
            #print(v[0],v[1])
            if v[1]!=u: #avoiding self loop conditions
                tmp+=G[u][v[1]]['act_prob']
        d[u]=tmp
        dd[u]=d[u]
        t[u]=0
    
    print("Running time precomputation:--- %s seconds ---" % (time.perf_counter() - start_time))
    num_seeds=0
    node_lookup=0
    while len(S0)<Max_trans_nodes and num_seeds<k:
        s=choose_next_s(all_nodes,dd,S0,cur)
        cur[s[1]]=True
        out_deg=len(list(G.out_edges(s[0])))
        node_lookup+=1
        #print('top deg-ic:',s[0],s[1],s[2])
        if s[0] in nodes_transmission:
            S0.append(s[0])
            S0_avg.append(s[2])
            num_seeds+=1
            print("chosen %d th node=%d; ddv[%d]=%f, out_edges=%f" % (num_seeds,s[0],s[1],s[2],out_deg))
        for child in G.out_edges(s[0]):
            #print(v[0],v[1],v[2])
            v=child[1]
            if v not in S0 and v!=s[0]: #v!=s[0] is to avoid self loop if present
                t[v]+=G[s[0]][v]['act_prob']
                dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
        
                
    val,failure=ic_relay(G, S0, path,m)
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
    strseeds=[nodes_map[s] for s in S0]
    print("The final best k critical transmission nodes after %d search are:"%node_lookup)
    print(strseeds,S0)
    #print('degree-discount in the network given S0:',S0_avg)
    print('final ic failures given S0:',failure)
    if writeFile:
        result=open(outdir+outfile+'_results_ddic_seeds_'+str(k)+'.txt','w')
        result.writelines(["%s\n" % item  for item in strseeds])
        result.close()
    
        result2=open(outdir+outfile+'_results_ddic_idx_seeds_'+str(k)+'.txt','w')
        result2.writelines(["%s\n" % str(item)  for item in S0])
        result2.close()
        
        with open(outdir+outfile+'_ddv.pkl', 'wb') as f:
            pickle.dump(dd, f)
        with open(outdir+outfile+'_tv.pkl', 'wb') as f:
            pickle.dump(t, f)
        with open(outdir+outfile+'_ic_failure.pkl', 'wb') as f:
            pickle.dump(val, f)
'''

def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def evaluate_seeds_2_hop(filedir,outdir,outfile,k,intermed_hop,
                         critical_fac='Military_Bases'):
    seed_file=open(outdir+outfile+'_results_ddic_seeds_'+str(k)+'.txt','r')
    seeds=[item.strip('\n') for item in seed_file]
    print('total_seeds:',len(seeds))
    
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',\
                              index_col=False,low_memory=False)
       
    #selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]
    
    #substations=pd.read_csv(infile+'_Electric_Substations.node',delimiter=',',index_col=False,low_memory=False)
    #print('substations info:',substations.shape)
    trans_To_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    interim_file='_Electric_Substations-'+intermed_hop+'.edge'
    sub_hop2_edge= pd.read_csv(infile+'urbannet2020-graph-v9/'+interim_file,\
                               delimiter=',',header=None,names=['u','v'],\
                               index_col=False)
    
    hop2_critical_edge=pd.read_csv(infile+'urbannet2020-graph-v9/'+\
                                  '_'+intermed_hop+'-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    hop2_To_critical=hop2_critical_edge['u'].drop_duplicates().values
    print(intermed_hop+'# connected to critical facility:',len(hop2_To_critical))
    
    sub_To_hop2=sub_hop2_edge[sub_hop2_edge['v'].isin(hop2_To_critical)]
    sub_To_hop2=sub_To_hop2['u'].drop_duplicates().values
    
    print('# substations connected to hop2:',len(sub_To_hop2))
    
    trans_To_sub=trans_To_sub[trans_To_sub['u'].isin(seeds)]
    trans_To_sub=trans_To_sub[trans_To_sub['v'].isin(sub_To_hop2)]
    
    final_critical_transmissions=trans_To_sub['u'].drop_duplicates().values
        
    true_critical_data=transmissions[transmissions['NODE_ID'].isin(final_critical_transmissions)]
    print('#critical transmission connected to critical facility in 3 hops:'\
          ,len(final_critical_transmissions),true_critical_data.shape)
    
    print(final_critical_transmissions)
    #true_critical_data.to_csv(outdir+outfile+'_true_critical_nodes_hop3_'+\
     #                         intermed_hop+'_'+critical_fac+'.csv',index=False)
        
def evaluate_seeds_1_hop(filedir,outdir,outfile,k,critical_fac='military'):
    seed_file=open(outdir+outfile+'_results_ddic_seeds_'+str(k)+'.txt','r')
    seeds=[item.strip('\n') for item in seed_file]
    print('total_seeds:',len(seeds))
    #print(seeds)
    
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False,low_memory=False)
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]   
    #print(selected_seed_data.shape)
    #print('seed voltages:')
    #print(selected_seed_data[['NODE_ID','VOLTAGE']])
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',delimiter=',',index_col=False,low_memory=False)
    print('substations info:',substations.shape)
    
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    #sub_military=pd.read_csv(infile+'_Electric_Substations-Military_Bases.edge'\
     #                     ,delimiter=',',header=None,names=['u','v'],index_col=False)
        
    sub_military=pd.read_csv(infile+'_Electric_Substations-Hospitals.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    sub_in_military=sub_military['u'].drop_duplicates().values
    print('#substations connected to critical facility:',len(sub_in_military))
    
    trans_sub=trans_sub[trans_sub['u'].isin(seeds)]
    
    sub_with_seeds=trans_sub['v'].drop_duplicates().values
    
    print('#substations connected to seeds:',len(sub_with_seeds))
    
    #get all those transmission lines whose substations are connected to military base
    #1. collect all the subs that connected to seeds and military
    seeded_subs_military=sub_military[sub_military['u'].isin(sub_with_seeds)]
    print('#seed-connected substations connected to critical facility:',seeded_subs_military.shape)
    #print(seeded_subs_military)
    seeded_subs_in_military=seeded_subs_military['u'].drop_duplicates().values
    print('#critical substations with critical facility:',len(seeded_subs_in_military))
    #get connected substation loads
    seeded_loads=substations[substations['NODE_ID'].isin(seeded_subs_in_military)]
    #seeded_loads=substations[substations['NODE_ID'].isin(sub_in_military)]
    #print('seeds critical loads:')
    #print(seeded_loads[['NODE_ID','LOAD_MW']])
    #collect all the transmission lines that are supplying to military
    trans_in_military=trans_sub[trans_sub['v'].isin(seeded_subs_in_military)]
    
    print('actual critical transmissions',trans_in_military.shape)
    
    trans_in_military=trans_in_military['u'].drop_duplicates().values
    print(trans_in_military)  
    
    true_critical_data=transmissions[transmissions['NODE_ID'].isin(trans_in_military)]
    
    selected_seed_data.to_csv(outdir+outfile+'_seeds_50_data.csv',index=False)
    true_critical_data.to_csv(outdir+outfile+'_true_critical_nodes_'+\
                              critical_fac+'.csv',index=False)
    seeded_loads.to_csv(outdir+outfile+'_critical_substations_'+critical_fac+'.csv'\
                        ,index=False)
    
if __name__=='__main__':
    filedir='data/' #sys.argv[1]
    outdir='output/no_domain_pj/'#sys.argv[2]
    #outdir='output/domain_based_pj/'
    outfile='urbannet_no_rule' #sys.argv[3] 
    #read_whole_graph(filedir,consumer1)
    m=10#sys.argv[4]
    k=50#sys.argv[5]
    #alpha: hyperparam choose stress increase on a node when its co-parent fails
    #alpha=0.2 #sys.argv[5]  
    
    p=0.01 #sys.argv[7]
    #outfile='toy'
    #edgefile='all_edges_rule_based_pj'
    edgefile='all_edges_no_rule_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    critical_fac=['Military_Bases','Hospitals']
    degreeDiscountIC(filedir,outdir,outfile,edgefile,k,p,m)
    evaluate_seeds_1_hop(filedir,outdir,outfile,k,critical_fac[0])
    evaluate_seeds_1_hop(filedir,outdir,outfile,k,critical_fac[1])
    
    '''
    intermed_military=['Aircraft_Landing_Facilities','Fire_Stations',\
                       'GeoTel_SupportingCOs','Hospitals',\
                       'Local_Law_Enforcement_Locations','Wastewater_Treatment_Plants']
    
    intermed_hospitals=['Fiber_Routes','Fire_Stations','Pharmacies',\
                        'GeoTel_SupportingCOs',\
                        'SWPA_DigitalMicrowaveLinks','SWPA_OpticalGroundWire',\
                        'Wastewater_Treatment_Plants',\
                        'EPA_Toxic_Release_Inventory_Facilities']

    for critical in critical_fac:
        for intermed in intermed_hospitals:
            print(intermed,critical_fac[1])
            evaluate_seeds_2_hop(filedir,outdir,outfile,k,intermed,critical_fac=critical)
    '''
    #seeds=[5,7]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
    #seeds=[11,13]
    #force_seed(filedir,outdir,k,consumer1,seeds,alpha,m)
        