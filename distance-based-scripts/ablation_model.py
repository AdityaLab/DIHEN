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
import preprocess_graph as prepg
import csv
import pandas as pd
from independent_cascade import independent_cascade
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
    
    #print('tl_v>=345,tes_v>=138,tl_v>=345+tes_v>=138,tl_v==345+tes_v>=138')
    #print(len(seed_345),len(tes_138),len(critical_tes_345_138),len(critical_tes_e345_138))
    #num_critical_tes=len(critical_tes)
    #print(critical_tes)
    
    critical_tes_data=trans_es[trans_es['NODE_ID'].isin(critical_tes)]
    print('total connected transmission substations:',critical_tes_data.shape[0])
    #critical_tes_data.to_csv(outdir+reg+'_critical_tes.csv',index=False)
     
    
def evaluate_seeds_1_hop(transmissions,substations,trans_sub,seeds,
                         k,critical_fac,outdir=None):
    
    infile='data/v9/'
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]   
    if outdir is not None:
        selected_seed_data.to_csv(outdir+'topk_transmission_lines.csv',index=False)
    seed_vol_match=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    vol_match=len(seed_vol_match.drop_duplicates().values)
    seed_vol_match=seed_vol_match['NODE_ID']
    
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
    
    return strseeds,S0,failure_spread,node_lookup
    
def get_act_prob(data,nodes_map,b1,check_rule):
    #check_rule==True, use b2 for edges transmission->substation
    #check_rule==False, use b2 for edges not transmission->substation
    act_prob=[]
    for ix,row in data.iterrows():
        #print(row['u'],row('v'))
        n1,n2=row['u'],row['v']
        u,v=nodes_map[n1],nodes_map[n2]
        tu,tv=u.split(':')[0],v.split(':')[0]
        if tu=='Transmission_Lines' and tv=='Electric_Substations'\
            or tv=='Transmission_Electric_Substations':
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
            
def check_ablation_model(filedir,outdir,outfile,edgefile,reg,K,p,m,critical_fac,cases):
    infile='../data/v9/'
    #path=filedir+'new_preprocessed_data/'
    path=filedir
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    robust_file=open(outdir+outfile+'_robustness.csv','w')
    robust_file.writelines('k,spread,lookup,c1_vol,c1_ruleb,c1_rule,c2_vol_match,c2_ruleb,c2_rule,total_rule\n')
    #K=[25,50,100,200,800,1600,3200,5000]
    print('Begin:')
    start_time = time.perf_counter()
    #prepg.read_whole_graph(filedir,consumer1) #preprocess urbannet graph
    nodes_file=open(path+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes_'+reg+'.txt')
    #df = pd.read_csv(path+edgefile+'.txt',delimiter=',',names=['u','v','kij','pj','p'])
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    
    #data_pij=pd.DataFrame()
    df['act_prob']=df.kij*(1-df.pj)
    edges=df['act_prob'].values
    mu=np.mean(edges)
    sigma=np.var(edges)
    total_edges=len(edges)
    print('edge-weight P distribution:')
    print('min, max:',np.min(edges),np.max(edges))
    print('mean, var:',mu,sigma)
    
    '''
    print('K distribution:')
    print('min, max:',df['kij'].min(),df['kij'].max())
    print('mean, var:',df['kij'].mean(),df['kij'].var())
    
    print('U distribution:')
    print('min, max:',df['pj'].min(),df['pj'].max())
    print('mean, var:',df['pj'].mean(),df['pj'].var())
    '''
    
    for case in cases:
            print('case:',case)
            tmp_data=df.copy()
            if case==1:#check effect of P 
                tmp_data['act_prob']=df.kij*(1-df.pj)
            elif case==2: #check effect of U
                tmp_data['act_prob']=1-df.pj
            elif case==3: #check effect of K
                tmp_data['act_prob']=df.kij
            elif case==4: #check effect of U for no domain rule
                tmp_data['act_prob']=get_act_prob(df,nodes_map,1,False)
            elif case==5: #check effect random edge weights with sampled from mean and var P
                random_weight=np.random.normal(mu, sigma, total_edges)
                tmp_data['act_prob']=random_weight
            #col='fij'
            #data_pij[col]=tmp_data['act_prob'].values
            
            seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,K[0],p,m)
            
            evaluate_connected_tes(outdir,transmissions,trans_es,seed_list,K[0],reg)
            
            true_seed_c1,c1_rulea_match,c1_ruleb_match,c1_num_rule,\
            c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                        substations,trans_sub,seed_list,K[0],critical_fac[0])
            
            true_seed_c2,c2_rulea_match,c2_ruleb_match,c2_num_rule,\
            c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                    trans_sub,seed_list,K[0],critical_fac[1])
            
            total_critical_seed=list(set(true_seed_c1).union(true_seed_c2))
            total_rule=len(total_critical_seed)
            
            print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
            print(len(seed_list),final_spread,lookup)
            print(c1_rulea_match,c1_ruleb_match,c1_num_rule)
            print(c2_rulea_match,c2_ruleb_match,c2_num_rule,total_rule)
            
            #string1=str(k)+','+str(b1)+','+str(b2)+','+str(final_spread)+','+str(lookup)+','
            string1=str(K[0])+','+str(final_spread)+','+str(lookup)+','
            string2=str(c1_rulea_match)+','+str(c1_ruleb_match)+','+str(c1_num_rule)+','
            string3=str(c2_rulea_match)+','+str(c2_ruleb_match)+','+\
            str(c2_num_rule)+','+str(total_rule)+'\n'
            
            robust_file.writelines(string1+string2+string3)
            '''
            with open(outdir+outfile+'_true_seed_c1_'+str(b1)+'_'+str(k)+'.txt','w') as res:
                res.writelines(["%s\n" % item  for item in true_seed_c1])
            
            with open(outdir+outfile+'_true_seed_c2_'+str(b1)+'_'+str(k)+'.txt','w') as res:
                res.writelines(["%s\n" % item  for item in true_seed_c2])
        
            '''
    #data_pij.to_csv(outdir+outfile+'_pij_values.csv',index=False)
    robust_file.close()
    
def check_network_scenario(filedir,outdir,outfile,edgefile,k,p,m,critical_fac,reg):
    infile='data/v9/'
    transmissions=pd.read_csv(infile+'_Transmission_Lines.node',\
                              delimiter=',',index_col=False,low_memory=False)
    
    substations=pd.read_csv(infile+'_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_es=pd.read_csv(infile+'_Transmission_Electric_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Lines-Electric_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    
    robust_file=open(outdir+outfile+'_scenario.csv','w')
    robust_file.writelines('reg,spread,lookup,c1_vol,c1_ruleb,c1_rule,c2_vol_match,c2_ruleb,c2_rule,total_rule\n')
    print('Begin:')
    start_time = time.perf_counter()
    
        
    nodes_file=open(filedir+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(filedir+'transmission_nodes_'+reg+'.txt')
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
    df['act_prob']=df.kij*(1-df.pj)
        
    seed_list,seed_idx,final_spread,lookup = \
        degreeDiscountIC(filedir,df,nodes_transmission,nodes_map,k,p,m)
    
    evaluate_connected_tes(outdir,transmissions,trans_es,seed_list,k)
    
    true_seed_c1,c1_rulea_match,c1_ruleb_match,c1_num_rule,\
        c1_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,
                    substations,trans_sub,seed_list,k,critical_fac[0],outdir=outdir)
            
    true_seed_c2,c2_rulea_match,c2_ruleb_match,c2_num_rule,\
        c2_num_seed_to_sub=evaluate_seeds_1_hop(transmissions,substations,\
                                trans_sub,seed_list,k,critical_fac[1])
            
    total_critical_seed=list(set(true_seed_c1).union(true_seed_c2))
    total_rule=len(total_critical_seed)
            
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
    print(c1_rulea_match,c1_ruleb_match,c1_num_rule)
    print(c2_rulea_match,c2_ruleb_match,c2_num_rule)
            
    string1=reg+','+str(final_spread)+','+str(lookup)+','
    string2=str(c1_rulea_match)+','+str(c1_ruleb_match)+','+str(c1_num_rule)+','
    string3=str(c2_rulea_match)+','+str(c2_ruleb_match)+','+\
            str(c2_num_rule)+','+str(total_rule)+'\n'
            
    robust_file.writelines(string1+string2+string3)
    with open(outdir+outfile+'_true_seed_c1_graph_'+reg+'.txt','w') as res:
        res.writelines(["%s\n" % item  for item in true_seed_c1])
            
    with open(outdir+outfile+'_true_seed_c2_graph_'+reg+'.txt','w') as res:
        res.writelines(["%s\n" % item  for item in true_seed_c2])
       
    
if __name__=='__main__':
    filedir=sys.argv[1]
    outdir='output/ablation_results/'#sys.argv[2]
   
    outfile=sys.argv[2] #'ablation_TX'
    m=10
    K=[50,100,200,300,500]#sys.argv[5]
    #K=[50]#sys.argv[5]
    
    p=0.1 #sys.argv[7]
    edgefile='TX_all_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases','Hospitals']
    region= sys.argv[3] #'TX'
    cases=[1,2,3,4,5]
    check_ablation_model(filedir,outdir,outfile,edgefile,region,K,p,m,critical_fac,cases)
    
