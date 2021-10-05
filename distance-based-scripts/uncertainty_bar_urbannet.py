#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 06:34:55 2021

@author: anikat
"""
import sys
import os
import numpy as np
import networkx as nx
import time
import csv
import pandas as pd
from independent_cascade import independent_cascade

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
    maxddv=-10000  
    for i in range(0,len(nodes)):
        if i==0 and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
        elif delta[nodes[i]] > maxddv and cur[i]==False: #and nodes[i] not in S0
            maxddv=delta[nodes[i]]
            ddv=nodes[i]
            maxi=i
    return [ddv,maxi,maxddv]

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

def ic_relay(G,seeds,filename,m,pre_computed_p):
    #S: failed seed nodes that have been selected so far
    #m: # simulation to run
    
    spread = 0
    #Maxnodes=max(list(G.nodes))
    val={}
    for node in list(G.nodes):
        val[node]=0
    val[0]=0
    for mc in range(0,m):
        H = independent_cascade(G,seeds,filename,pre_computed_p)
        for i in range(0,len(H)):
            for j in range(0,len(H[i])):
                if H[i][j] in list(G.nodes):
                    val[H[i][j]]+=1
    for i in val.keys():
        val[i]=val[i]/m
        spread+=val[i]
    return val,spread

def evaluate_seeds_2_hop(transmissions,substations,trans_sub,seeds,
                         k,critical_fac,outdir=None):
    
    infile='../data/v9/'
    selected_seed_data=transmissions[transmissions['NODE_ID'].isin(seeds)]   
    if outdir is not None:
        selected_seed_data.to_csv(outdir+'topk_transmission_lines.csv',index=False)
    seed_vol_match=selected_seed_data.loc[selected_seed_data['VOLTAGE'] >= 345]
    r1=len(seed_vol_match.drop_duplicates().values)
    seed_vol_match=seed_vol_match['NODE_ID']
    sub_critical=pd.read_csv(infile+'_Electric_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    
    trans_sub=trans_sub[trans_sub['u'].isin(seeds)]
    
    sub_with_seeds=trans_sub['v'].drop_duplicates().values
    #num_seed_to_sub=len(sub_with_seeds)
    
    #get all those transmission lines whose substations are connected to military base
    seeded_subs_military=sub_critical[sub_critical['u'].isin(sub_with_seeds)]
    seeded_subs_in_military=seeded_subs_military['u'].drop_duplicates().values
    #collect all the transmission lines that are supplying to military
    trans_in_military=trans_sub[trans_sub['v'].isin(seeded_subs_in_military)]
    
    trans_in_military=trans_in_military['u'].drop_duplicates().values
    r2=len(trans_in_military)
    
    true_critical_data=transmissions[transmissions['NODE_ID'].isin(trans_in_military)]
    true_critical_data=true_critical_data.loc[true_critical_data['VOLTAGE'] >= 345]
    
    r3=len(true_critical_data)
    
    return r1,r2,r3

def degreeDiscountIC(path,data,nodes_transmission,nodes_map,k,p,m):
    G=nx.from_pandas_edgelist(data, 'u', 'v',edge_attr=['pnew','act_prob'],create_using=nx.DiGraph())
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
            if v[1]!=u and float(G[u][v[1]]['pnew'])<G[u][v[1]]['act_prob']: #avoiding self loop conditions
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
                if float(G[s[0]][v]['pnew'])<G[s[0]][v]['act_prob']: 
                    ##t[v]+=G[s[0]][v]['act_prob']
                    t[v]+=1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
            
    val,failure_spread=ic_relay(G, S0, path,m,False)
    strseeds=[nodes_map[s] for s in S0]
    
    return strseeds,S0,failure_spread,node_lookup

def check_uncertainty(filedir,outdir,outfile,edgefile,reg,K,p,m,critical_fac,iteration,case):
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
    total_nodes=118956
    original_R=np.array([1660,4201,99])/total_nodes
    
    ufile=open(outdir+outfile+'_'+str(K[0])+'_uncertianty_urbannet.csv','w')
    ufile.writelines('model_name,it,k,spread,lookup,R1,R2,R3,R1_g,R2_g,R3_g\n')
    print('Begin:')
    start_time = time.perf_counter()
    nodes_file=open(path+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes_'+reg+'.txt')
     
    df = pd.read_csv(filedir+reg+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])  
    df['act_prob']=df.kij*(1-df.pj)
    total_edges=df.shape[0]
    model_name='DIHeN'
    if case==2: #check effect of U
        df['act_prob']=1-df.pj
        model_name='CB'
    elif case==3: #check effect of K
        df['act_prob']=df.kij
        model_name='FP'
    elif case==4: #check effect of U for no domain rule
        df['act_prob']=get_act_prob(df,nodes_map,1,False)
        model_name='NER'
    elif case==5: #check effect random edge weights with sampled from mean and var P
        edges=df['act_prob'].values
        mu=np.mean(edges)
        sigma=np.var(edges)
    
        random_weight=np.random.normal(mu, sigma, total_edges)
        df['act_prob']=random_weight
        model_name='RG'
    
    for it in range(0,iteration):  
        tmp_data=df.copy()
        tmp_data['pnew']=np.random.uniform(0,1,total_edges)
        seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,K[0],p,m)
            
        r1,r2,r3=evaluate_seeds_2_hop(transmissions,
                        substations,trans_sub,seed_list,K[0],critical_fac[0])
            
        r1_g=(r1/K[0])/original_R[0]
        r2_g=(r2/K[0])/original_R[1]
        r3_g=(r3/K[0])/original_R[2]
            
        #print(it,len(seed_list),final_spread,lookup)
        print(it,r1,r2,r3)
            
        string1=model_name+','+str(it)+','+str(len(seed_list))+','+str(final_spread)+','+str(lookup)+','
        string2=str(r1)+','+str(r2)+','+str(r3)+','
        string3=str(r1_g)+','+str(r2_g)+','+str(r3_g)+'\n'
            
        ufile.writelines(string1+string2+string3)
            
    ufile.close()
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))

if __name__=='__main__':
    filedir=sys.argv[1] #'../data/G_for_robustness/'
    outfile= sys.argv[2] #'uncertain_national'
    outdir='../output/uncertainty_bar_urbannet/'#sys.argv[2]
    m=10#sys.argv[4]
    region= sys.argv[3] #'national'
    k= int(sys.argv[4])
    case=int(sys.argv[5])
    K=[k]
    
    p=0.1 #sys.argv[7]
    edgefile='_all_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Military_Bases']
    iteration=1000
    
    check_uncertainty(filedir,outdir,outfile,edgefile,region,K,p,m,critical_fac,iteration,case)
