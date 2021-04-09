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
import preprocess_full_urbannet as prepg
import csv
import pandas as pd
from independent_cascade import *
import pickle

#sys.stdout = open('simulation_run_urbannet_ddic.txt','w')
def _get_array(filename):
    with open(filename,'r') as f:
        arrayt = []
        for line in f:
            arrayt.append(int(line))
    #for i in range(0,len(arrayt)):
     #   arrayt[i]=int(arrayt[i])
    return arrayt

def draw_plot_k(k,spread):
    import matplotlib.pyplot as plt
    plt.plot(k,spread)
    plt.show()
    
def choose_next_s(nodes,delta,S0,cur):
    '''
    this part is to choose the node with max degree-discount gain
    '''
    #set min marginal gain be a smaller number, so any results will be chosen to update mind
    maxddv=-1000  
    for i in range(0,len(nodes)):
        #print("i=%d" % i)
        if i==0 and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
        elif delta[nodes[i]] > maxddv and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
    #print("mins=%d,i=%d" % (mins,mini))
    return [ddv,maxi,maxddv]

def ic_relay(G,seeds,filename,m,alpha=0.2,pre_computed_p=False):
    #S: failed seed nodes that have been selected so far
    #alpha: hyperparamaeter to select stress increase on uncertainty
    #m: # simulation to run
    
    spread = 0
    Maxnodes=max(list(G.nodes))
    #val=[0]*(Maxnodes+1)
    val={}
    for node in list(G.nodes):
        val[node]=0
    val[0]=0
    test=0
    #print('chosen seeds:',seeds)
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,alpha,pre_computed_p,steps=0)
        print("IC run:"+str(mc))
        #print(H)
        #time.sleep(5)
        #__all__ = ['independent_cascade']
        for i in range(0,len(H)):
            for j in range(0,len(H[i])):
                if H[i][j] in list(G.nodes):
                    val[H[i][j]]+=1
    for i in val.keys():
        #val[i]=1-val[i]/m
        val[i]=val[i]/m
        spread+=val[i]
    #print('val',val)
    #print('spread ',spread)
    return val,spread

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

def degreeDiscountIC(indir,outdir,outfile,edgefile,k,p,m=10):
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file,delimiter=' ')
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes.txt')
    df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
    G=nx.from_pandas_edgelist(df,'u','v',edge_attr=['act_prob','p'],create_using=nx.DiGraph())
    
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
            if v[1]!=u and float(G[u][v[1]]['p'])<G[u][v[1]]['act_prob']: #avoiding self loop conditions
                ##tmp+=G[u][v[1]]['act_prob']
                tmp+=1
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
                if float(G[s[0]][v]['p'])<G[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    print("The final best k critical transmission nodes after %d search are:"%node_lookup)    
    print("Running time to find k critical nodes:--- %s seconds ---" % (time.perf_counter() - start_time))
    strseeds=[nodes_map[s] for s in S0]
    with open(outdir+outfile+'_ddv.pkl', 'wb') as f:
        pickle.dump(dd, f)
    with open(outdir+outfile+'_tv.pkl', 'wb') as f:
        pickle.dump(t, f)
    
    result=open(outdir+outfile+'_results_seeds_'+str(k)+'.txt','w')
    result.writelines(["%s\n" % item  for item in strseeds])
    result.close()
    
    result2=open(outdir+outfile+'_results_idx_seeds_'+str(k)+'.txt','w')
    result2.writelines(["%s\n" % str(item)  for item in S0])
    result2.close()
    #print(strseeds,S0)
    #print('degree-discount in the network given S0:',S0_avg)
    
    
    '''
    S0=[]
    with open(outdir+outfile+'_results_idx_seeds_'+str(k)+'.txt', 'r') as res:
        for line in res: 
            line = line.strip('\n') #or some other preprocessing
            S0.append(int(line)) 
    '''
    topK=[5,10,20,30,50,70,100]
    for k in topK:          
        val,failure=ic_relay(G, S0[:k], path,m,pre_computed_p=True)
        print('final ic failures given k= %d : %f' %(k,failure))
        print("IC running time:--- %s seconds ---" % (time.perf_counter() - start_time))
        strfailure=[]
        for n in val.keys():
            if val[n]>0:
                strfailure.append(nodes_map[n])
        with open(outdir+outfile+'_ic_failure_top_'+str(k)+'.txt', 'w') as f:
            for item in strfailure:
                f.writelines("%s\n" %item)
            #pickle.dump(val, f)
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))

def check_page_rank_effect(outfile,outdir,k,critical_fac):
    infile='data/v9/'
    nodes_file=open(infile+'urbannet2020-graph-v9/index_seq.txt','r')
    nodes_list=csv.reader(nodes_file,delimiter=' ')
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    
    crscore_file=open(outdir+'sort0-10.txt','r')
    score_list=csv.reader(crscore_file,delimiter=' ')
    
    seeds=[int(row[0]) for row in score_list]
    num=0
    seed_list=[]
    for s in seeds:
        node=nodes_map[s]
        if node.split(':')[0]=='Transmission_Lines':
            seed_list.append(node)
            num+=1
        if num>=k:
            break
    
    reg='national'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
        
    evaluate_connected_tes(outdir,transmissions,trans_es,seed_list,k,reg)
    true_seed_c1,rulesc1,c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list,k,critical_fac[0],reg,True,outdir=outdir)
            
    true_seed_c2,rulesc2,c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list,k,critical_fac[1],reg,False)
            
    total_critical_seed=true_seed_c1+true_seed_c2
    #total_critical_seed1=set(total_critical_seed)
    total_rule=len(total_critical_seed)
    rulesc2['total_rule']=total_rule   
    #print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    print('total nodes:',len(list(nodes_map.keys())))
    print(rulesc1)
    print(rulesc2)

def scenario_winterstorm(filedir,outfile,outdir,k,critical_fac):
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    path=filedir+'preprocessed_data/'
    nodes_file=open(path+'all_nodes_index.txt','r')
    nodes_list=csv.reader(nodes_file,delimiter=',')
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
    G=nx.from_pandas_edgelist(df,'u','v',edge_attr=['act_prob','p'],create_using=nx.DiGraph())
    
    print('Total nodes:',len(G.nodes()))
    print('Total edges:',len(G.edges()))
    
    #seed_file=open(outdir+outfile+'ng_pipelines.txt','r')
    S0=[496225,496478,496540,496569,497032,497288,497523,497588,498251,499667,499679,499801,499802,499803,499804,
        499805,499844,499845,501688,502785,503305,503317,504163,505010,523496,508463,508836,511213,512093,513422,512093,513422,511119]
    m=10
    val,failure=ic_relay(G, S0, path,m,pre_computed_p=False)
    print("IC running time:--- %s seconds ---" % (time.perf_counter() - start_time))
    strfailure=[]
    for n in val.keys():
        if val[n]>0:
            strfailure.append(nodes_map[n])
    with open(outdir+outfile+'_ic_failure_ng_pipeline.txt', 'w') as f:
        for item in strfailure:
            f.writelines("%s\n" %item)
    print("Total Running time:--- %s seconds ---" % (time.perf_counter() - start_time))
    
def check_effect(outfile,outdir,k,critical_fac):
    seed_file=open(outdir+outfile+'_results_seeds_'+str(k)+'.txt','r')
    
    seeds=[item.strip('\n') for item in seed_file]
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
        
    true_seed_c1,c1_rulea_match,c1_ruleb_match,c1_num_rule,\
            c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                        substations,trans_sub,seeds,k,critical_fac[0])
            
    true_seed_c2,c2_rulea_match,c2_ruleb_match,c2_num_rule,\
        c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                    trans_sub,seeds,k,critical_fac[1])
            
    total_critical_seed=list(set(true_seed_c1).union(true_seed_c2))
    total_rule=len(total_critical_seed)
            
            
    print(c1_rulea_match,c1_ruleb_match,c1_num_rule)
    print(c2_rulea_match,c2_ruleb_match,c2_num_rule)
    print('total_rule:',total_rule)
    
if __name__=='__main__':
    filedir='data/v9/urbannet2020-graph-v9/' #sys.argv[1]
    #consumer1='Natural_Gas_Compressor_Stations'#sys.argv[2]
    outdir='output/urbannet-whole-network/'#sys.argv[3]
    outfile='TX_winterstorm' #sys.argv[4] 
    m=1#sys.argv[5]
    k=100#sys.argv[6]
    
    p=0.01 #sys.argv[7]
    
    edgefile='urbannet_edges_rule_based_pj_p' #sys.argv[8]
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    #prepg.uncertainty_with_domain(filedir,edgefile)
    
    #degreeDiscountIC(filedir,outdir,outfile,edgefile,k,p,m)
    
    critical_fac=['Military_Bases','Hospitals']
    #check_effect(outfile,outdir,k,critical_fac)
    check_page_rank_effect(outfile,outdir,k,critical_fac)
    #scenario_winterstorm(filedir,outfile,outdir,k,critical_fac)
    '''
    evaluate_seeds_1_hop(filedir,outdir,outfile,k,critical_fac[1])
    k=[5,10,20,30,50,70,100]
    spread=[398.2,590.2,822,1074.8,1496.3,1798.9,2194.7]
    draw_plot_k(k,spread)
    '''
    '''
    intermed_military=['Aircraft_Landing_Facilities','Fire_Stations',\
                       'GeoTel_SupportingCOs','Hospitals',\
                       'Local_Law_Enforcement_Locations','Wastewater_Treatment_Plants']
    
    intermed_hospitals=['Fiber_Routes','Fire_Stations','Pharmacies',\
                        'GeoTel_SupportingCOs',\
                        'SWPA_DigitalMicrowaveLinks','SWPA_OpticalGroundWire',\
                        'Wastewater_Treatment_Plants',\
                        'EPA_Toxic_Release_Inventory_Facilities']
    
    for intermed in intermed_hospitals:
        print(intermed,critical_fac[1])
        evaluate_seeds_2_hop(filedir,outdir,outfile,k,intermed,critical_fac=critical_fac[1])
    '''
        