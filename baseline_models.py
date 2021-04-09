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

def ic_relay(G,seeds,filename,m,pre_computed_p):
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
        H = independent_cascade(G,seeds,filename,pre_computed_p,steps=0)
        #print("IC run:"+str(mc))
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

def evaluate_connected_tes(outdir,transmissions,trans_es,seeds,k,reg):
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]
    
    filename1='_Transmission_Lines-Transmission_Electric_Substations.edge'
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
                         k,critical_fac):
    
    infile='data/v9/'
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]   
    #print('seed voltages:')
    #print(selected_seed_data[['NODE_ID','VOLTAGE']])
    
    seed_vol_match=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    vol_match=len(seed_vol_match.drop_duplicates().values)
    seed_vol_match=seed_vol_match['NODE_ID']
    
    sub_critical=pd.read_csv(infile+'_Electric_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    #sub_in_critical=sub_critical['u'].drop_duplicates().values
    #num_sub_to_critical=len(sub_in_critical)
    
    trans_sub=trans_sub[trans_sub['u'].isin(seeds)]
    
    sub_with_seeds=trans_sub['v'].drop_duplicates().values
    num_seed_to_sub=len(sub_with_seeds)
    
    #get all those transmission lines whose substations are connected to military base
    #1. collect all the subs that connected to seeds and military
    seeded_subs_military=sub_critical[sub_critical['u'].isin(sub_with_seeds)]
    #print('#seed-connected substations connected to critical facility:',seeded_subs_military.shape)
    
    seeded_subs_in_military=seeded_subs_military['u'].drop_duplicates().values
    #print('#critical substations with critical facility:',len(seeded_subs_in_military))
    #num_sub_to_critical=len(seeded_subs_in_military)
    
    #collect all the transmission lines that are supplying to military
    trans_in_military=trans_sub[trans_sub['v'].isin(seeded_subs_in_military)]
    
    #print('actual critical transmissions',trans_in_military.shape)
    
    trans_in_military=trans_in_military['u'].drop_duplicates().values
    num_ruleb=len(trans_in_military)
    
    true_critical_data=transmissions[transmissions['NODE_ID'].isin(trans_in_military)]
    true_critical_data=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 345]
    
    #true_critical_nodes=true_critical_data['NODE_ID'].values
    num_rule1=len(true_critical_data)
    
    return trans_in_military,vol_match,num_ruleb,num_rule1,num_seed_to_sub
    
def degreeDiscountIC(path,data,nodes_transmission,nodes_map,k,p,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['p','act_prob'],create_using=nx.DiGraph())
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
        for v in G.out_edges(u):
            if v[1]!=u and float(G[u][v[1]]['p'])<G[u][v[1]]['act_prob']: #avoiding self loop conditions
                ##tmp+=G[u][v[1]]['act_prob']
                tmp+=1
        d[u]=tmp
        dd[u]=d[u]
        t[u]=0
    
    num_seeds=0
    node_lookup=0
    while len(S0)<Max_trans_nodes and num_seeds<k:
        s=choose_next_s(all_nodes,dd,S0,cur)
        cur[s[1]]=True
        #out_deg=len(list(G.out_edges(s[0])))
        node_lookup+=1
        #print('top deg-ic:',s[0],s[1],s[2])
        if s[0] in nodes_transmission:
            S0.append(s[0])
            S0_avg.append(s[2])
        num_seeds+=1
        #print("chosen %d th node=%d; ddv[%d]=%f, out_edges=%f" % (num_seeds,s[0],s[1],s[2],out_deg))
        for child in G.out_edges(s[0]):
            v=child[1]
            #v!=s[0] is to avoid self loop if present
            if v not in S0 and v!=s[0]:
                if float(G[s[0]][v]['p'])<G[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    val,failure_spread=ic_relay(G, S0, path,m,True)
    strseeds=[nodes_map[s] for s in S0]
    #print(strseeds,S0)
    #print('final ic failures given S0:',failure_spread)
    
    return strseeds,S0,failure_spread,node_lookup
    
def get_act_prob(data,pij):
    act_prob=[pij]*data.shape[0]
    
    return act_prob

def collect_seeds(path,G,val,nodes_map,k,m):
    from collections import OrderedDict
    sorted_val = OrderedDict(sorted(val.items(),key=lambda item: item[1],reverse=True))
    
    S0=[]
    num_seed=0
    for key in sorted_val.keys():  
        if num_seed>=k:
            break
        node_type=nodes_map[key].split(':')[0]
        if node_type=='Transmission_Lines':
            #print(key,sorted_val[key])
            S0.append(key)
        num_seed+=1
    
    strseeds=[nodes_map[s] for s in S0] 
    #val,failure_spread=ic_relay(G, S0, path,m,True)
    
    return strseeds, S0, -1

def page_rank(path,data,nodes_map,k,m,is_weighted=False):
    if is_weighted:
        G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
        pr= nx.pagerank(G, alpha=0.9,weight='act_prob')
    else:
        G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
        pr= nx.pagerank(G, alpha=0.9)    
    
    return collect_seeds(path,G,pr,nodes_map,k,m)

def critical_score(path,data,nodes_map,k,m,pr_score,rpr_score,is_weighted=False):
    if is_weighted:
        G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['act_prob'],create_using=nx.DiGraph())
        G_rev=nx.DiGraph.reverse(G)
        pr= nx.pagerank(G, alpha=0.9,weight='act_prob')
        r_pr=nx.pagerank(G_rev, alpha=0.9,weight='act_prob')
    else:
        G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
        G_rev=nx.DiGraph.reverse(G)
        pr= nx.pagerank(G, alpha=0.9)    
        r_pr=nx.pagerank(G_rev, alpha=0.9,weight='act_prob')
        
    cr_score={}
    for key in pr.keys():
        cr_score[key]=pr_score*pr[key]+rpr_score*r_pr[key]
        
    
    return collect_seeds(path,G,cr_score,nodes_map,k,m)

def degree_centrality_rank(path,data,nodes_map,k,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
    deg=nx.degree_centrality(G)
    
    return collect_seeds(path,G,deg,nodes_map,k,m)

def clustering(path,data,nodes_map,k,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',create_using=nx.DiGraph())
    cluster=nx.clustering(G)
    
    return collect_seeds(path,G,cluster,nodes_map,k,m)

def random_nodes(path,nodes_map,tnodes_all,k):
    import random
    nodes_tline=random.sample(tnodes_all, k)
    strseeds=[nodes_map[s] for s in nodes_tline] 
    return strseeds,nodes_tline,-1
        
def writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,\
              seed_list,k,pj,critical_fac,lookup,final_spread):
   
   evaluate_connected_tes(outdir,transmissions,trans_es,seed_list,k,'TX')
   true_seed_c1,c1_rulea_match,c1_ruleb_match,c1_num_rule,\
       c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                trans_sub,seed_list,k,critical_fac[0])
   
   true_seed_c2,c2_rulea_match,c2_ruleb_match,c2_num_rule,c2_num_seed_to_sub\
    =evaluate_seeds_1_hop(transmissions,substations,\
                trans_sub,seed_list,k,critical_fac[1])
   
   total_critical_seed=list(set(true_seed_c1).union(true_seed_c2))
   total_rule=len(total_critical_seed)
    
   print(c1_rulea_match,c1_ruleb_match,c1_num_rule)
   print(c2_rulea_match,c2_ruleb_match,c2_num_rule,total_rule)
         
   string1=pj+','+str(final_spread)+','+str(lookup)+','
   string2=str(c1_rulea_match)+','+str(c1_ruleb_match)+','+str(c1_num_rule)+','
   string3=str(c2_rulea_match)+','+str(c2_ruleb_match)+','+\
            str(c2_num_rule)+','+str(total_rule)+'\n'
   ''' 
   with open(outdir+outfile+'_true_seed_c1_'+pj+'.txt','w') as res:
            res.writelines(["%s\n" % item  for item in true_seed_c1])
            
   with open(outdir+outfile+'_true_seed_c2_'+pj+'.txt','w') as res:
            res.writelines(["%s\n" % item  for item in true_seed_c2])
   '''
   return string1+string2+string3
             
def check_baseline_model(filedir,outdir,outfile,k,p,m,P,critical_fac,reg):
    infile='data/v9/'
    path=filedir+'new_preprocessed_data/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
        
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    robust_file=open(outdir+outfile+'_robustness.csv','w')
    robust_file.writelines('pj,spread,lookup,c1_vol,c1_ruleb,c1_rule,c2_vol_match,c2_ruleb,c2_rule,total_rule\n')
    
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
    #nodes_file=open(filedir+'all_nodes_index_TX.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    #df = pd.read_csv(filedir+'TX_'+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
    
    
    print('critical score-10:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,1,0,is_weighted=False)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'critical-score10',critical_fac,len(seed_list),final_spread)
    #print(line)
    robust_file.writelines(line)
    
    print('critical score-01:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,0,1,is_weighted=False)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'critical-score01',critical_fac,len(seed_list),final_spread)
    #print(line)
    robust_file.writelines(line)
    
    print('critical score-55:')
    seed_list,seed_idx,final_spread=critical_score(path,df,nodes_map,k,m,0.5,0.5,is_weighted=False)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'critical-score55',critical_fac,len(seed_list),final_spread)
    robust_file.writelines(line)
    
    
    print('page rank unweighted:')
    seed_list,seed_idx,final_spread=page_rank(path,df,nodes_map,k,m)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'pagerank_unweight',critical_fac,-1,final_spread)
    robust_file.writelines(line)
    
    print('page rank weighted:')
    seed_list,seed_idx,final_spread=page_rank(path,df,nodes_map,k,m,is_weighted=True)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'pagerank_weighted',critical_fac,-1,final_spread)
    robust_file.writelines(line)
    
    print('degree-centrality:')
    seed_list,seed_idx,final_spread=degree_centrality_rank(path,df,nodes_map,k,m)
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,'deg-cent',critical_fac,-1,final_spread)
    robust_file.writelines(line)
    
    '''
    print('random:')
    tnodes_all=_get_array(path+'transmission_nodes.txt')
    for it in range(0,100):
        seed_list,seed_idx,final_spread=random_nodes(path,nodes_map,tnodes_all,k)
        line=writefile(outdir,outfile,transmissions,substations,trans_sub,seed_list,k,'random',critical_fac,it,final_spread)
        robust_file.writelines(line)
    '''
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    
    
    for pj in P: 
        print('diffusion--',pj)
        tmp_data=df.copy()
        tmp_data['act_prob']=get_act_prob(df,pj)
            
        seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,k,pj,m)
            
            
        print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
        
        line=writefile(outdir,outfile,transmissions,substations,trans_sub,trans_es,seed_list,k,str(pj),critical_fac,lookup,final_spread)
        robust_file.writelines(line)
        
    robust_file.close()
    
    
       
if __name__=='__main__':
    #filedir='data/' #sys.argv[1]
    filedir='data/G_for_robustness/' #for ercot baseline
    #outdir='output/baseline_results/'#sys.argv[2]
    outdir='output/baseline_results_all_data/'#sys.argv[2]
    outfile='baseline_tx' #sys.argv[3] 
    #read_whole_graph(filedir,consumer1)
    m=10#sys.argv[4]
    k=50#sys.argv[5]
    
    p=0.1 #sys.argv[7]
    #edgefile='national_all_edges_rule_based_pj_p'
    edgefile='TX_all_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases','Hospitals']
    
    #P=[1,0.9,0.7,0.5,0.3,0.2,0.1]
    P=[1,0.9,0.7,0.5,0.3,0.2]
    reg='TX'
    #check_baseline_model(filedir,outdir,outfile,k,p,m,P,critical_fac,reg,pr_sc=0,rpr_sc=1)
    #check_baseline_model(filedir,outdir,outfile,k,p,m,P,critical_fac,reg,pr_sc=0.5,rpr_sc=0.5)
    check_baseline_model(filedir,outdir,outfile,k,p,m,P,critical_fac,reg)
    
        