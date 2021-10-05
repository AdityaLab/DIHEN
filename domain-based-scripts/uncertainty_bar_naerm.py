#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:29:50 2021

@author: anikat
This script is to run ablation models 1000 times to compute error bars
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
    
    return arrayt

def get_act_prob(data,nodes_map,b1,check_rule):
    #check_rule==True, use b2 for edges transmission->substation
    #check_rule==False, use b2 for edges not transmission->substation
    act_prob=[]
    for ix,row in data.iterrows():
        #print(row['u'],row('v'))
        n1,n2=row['u'],row['v']
        u,v=nodes_map[n1],nodes_map[n2]
        tu,tv=u.split(':')[0],v.split(':')[0]
        if tu=='Transmission_Lines' and tv=='Distribution_Substations'\
            or tv=='Transmission_Substations':
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

def evaluate_R1(trans_sub,trans_line,seed_list,nodes_map,k):
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)] 
    dihen_trans_sub=selected_trans_sub[selected_trans_sub['NOMKVMAX']>=345.0]['NODE_ID'].values
    
    selected_trans_line=trans_line[trans_line['NODE_ID'].isin(seed_list)]
    dihen_trans_line=selected_trans_line[selected_trans_line['MAX_NOM_KV']>=345.0]['NODE_ID'].values
    
    dihen_seed=list(dihen_trans_sub)+list(dihen_trans_line)
    
    return len(dihen_seed)

def evaluate_R2R3(dis_sub,trans_sub,seed_list,k,critical_fac):
    infile='../data/naerm/edge-files/'

    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)]
    
    dis_critical_edge=pd.read_csv(infile+'_Distribution_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    trans_dis_edge= pd.read_csv(infile+'_Transmission_Substations-_Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    dis_seeds=selected_dis_sub['NODE_ID'].values
    trans_seeds=selected_trans_sub['NODE_ID'].values
    
    temp=trans_dis_edge[trans_dis_edge['u'].isin(trans_seeds)]
    dis_seeds2=temp['v'].drop_duplicates().values
    dis_seeds=list(dis_seeds)
    dis_seeds2=list(dis_seeds2)
    for s in dis_seeds2:
        if s not in dis_seeds:
            dis_seeds.append(s)
    
   
    selected_dis_military=dis_critical_edge[dis_critical_edge['u'].isin(dis_seeds)]
    selected_trans_for_dis=trans_dis_edge[trans_dis_edge['v'].isin(dis_seeds)]
    
    #trans_seeds2=selected_dis_for_trans['v'].drop_duplicates().values
    #selected_dis_military2=dis_critical_edge[dis_critical_edge['u'].isin(trans_seeds2)]
    
    selected_dis=selected_dis_military['u'].drop_duplicates().values
    selected_trans=selected_trans_for_dis['u'].drop_duplicates().values
    #print('#seed-connected substations connected to critical facility:',seeded_subs_military.shape)
    
    selected_r3=trans_sub[trans_sub['NODE_ID'].isin(selected_trans)]
    selected_r3=selected_r3.loc[selected_r3['NOMKVMAX']>=345.0]
    
    return len(selected_dis),len(selected_r3)

def evaluate_R7(trans_sub,dis_sub,seed_list,k,nodes_map):     
    infile='../data/naerm/edge-files/'
    
    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    dihen_dis=selected_dis_sub[selected_dis_sub['LOADMW']>=300.0]['NODE_ID'].values
    
    selected_trans_sub=trans_sub[trans_sub['NODE_ID'].isin(seed_list)]
    dihen_trans=selected_trans_sub[selected_trans_sub['LOADMW']>=300.0]['NODE_ID'].values
    
    dihen_seed=list(dihen_dis)+list(dihen_trans)
    
    return len(dihen_seed)

def evaluate_R8(trans_sub,dis_sub,trans_to_dis,seed_list,nodes_map,k,critical_fac):
    selected_dis_sub=dis_sub[dis_sub['NODE_ID'].isin(seed_list)]
    
    selected_dis_sub=selected_dis_sub['NODE_ID'].values
    
    infile='../data/naerm/edge-files/'
    edges=pd.read_csv(infile+'_Distribution_Substations-'+critical_fac+'.edge'\
                          ,delimiter=',',header=None,names=['u','v'])
    
    dihen_seed=edges[edges['u'].isin(selected_dis_sub)]
    dihen_seed=dihen_seed['u'].drop_duplicates().values

    return len(dihen_seed)


def check_uncertainty(path,outdir,outfile,edgefile,reg,K,p,m,critical_fac,iteration,case):
    infile='../data/naerm/edge-files/'
    trans_line=pd.read_csv(infile+'_Transmission_Lines.node',delimiter=',',index_col=False,low_memory=False)
    
    dis_sub=pd.read_csv(infile+'_Distribution_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_sub=pd.read_csv(infile+'_Transmission_Substations.node',\
                            delimiter=',',index_col=False,low_memory=False)
    
    trans_to_dis=pd.read_csv(infile+'_Transmission_Substations-_Distribution_Substations.edge'\
                          ,delimiter=',',header=None,names=['u','v'],index_col=False)
    
    
    total_nodes=93579
    original_R=np.array([874,3157,100,127,954])/total_nodes
    
    nodes_file=open(path+'all_nodes_index_'+reg+'.txt','r')
    nodes_list=csv.reader(nodes_file)
    nodes_map={int(row[1]):row[0] for row in nodes_list}
    nodes_transmission=_get_array(path+'transmission_nodes_'+reg+'.txt')
    
    df = pd.read_csv(filedir+edgefile+'.txt',delimiter=',',names=['u','v','kij','no_domain','pj','p'])
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
    
    ufile=open(outdir+outfile+'_uncertianty_naerm_'+model_name+'.csv','w')
    ufile.writelines('model_name,it,k,spread,lookup,R1,R2,R3,R7,R8,R1_g,R2_g,R3_g,R7_g,R8_g\n')
    print('Begin:')
    start_time = time.perf_counter()
    
    for it in range(0,iteration):  
        tmp_data=df.copy()
        tmp_data['pnew']=np.random.uniform(0,1,total_edges)
        seed_list,seed_idx,final_spread,lookup = \
            degreeDiscountIC(path,tmp_data,nodes_transmission,nodes_map,K[0],p,m)
        
        r1 = evaluate_R1(trans_sub,trans_line,seed_list,nodes_map,K[0])
        r2,r3 = evaluate_R2R3(dis_sub,trans_sub,seed_list,K[0],critical_fac[1])
        r7 = evaluate_R7(trans_sub,dis_sub,seed_list,K[0],nodes_map)
        r8 = evaluate_R8(trans_sub,dis_sub,trans_to_dis,seed_list,nodes_map,K[0],critical_fac[0])
            
        r1_g=(r1/K[0])/original_R[0]
        r2_g=(r2/K[0])/original_R[1]
        r3_g=(r3/K[0])/original_R[2]
        r7_g=(r7/K[0])/original_R[3]
        r8_g=(r8/K[0])/original_R[4]
        print(it,r1,r2,r3,r7,r8)
            
        string1=model_name+','+str(it)+','+str(len(seed_list))+','+str(final_spread)+','+str(lookup)+','
        string2=str(r1)+','+str(r2)+','+str(r3)+','+str(r7)+','+str(r8)+','
        string3=str(r1_g)+','+str(r2_g)+','+str(r3_g)+','+str(r7_g)+','+str(r8_g)+'\n'
            
        ufile.writelines(string1+string2+string3)
            
    ufile.close()
    print("Total Running time:%s seconds" % (time.perf_counter() - start_time))
            
             
if __name__=='__main__':
    filedir= sys.argv[1] #'../data/naerm/naerm_regional/'
    outdir='../output/uncertain_bounds/'#sys.argv[2]
   
    outfile='national' #sys.argv[3] 
    m=10#sys.argv[4]
    k= int(sys.argv[2]) #[500]
    case= int(sys.argv[3])
    K=[k]
    iteration= 1000
    p=0.1 
    edgefile='national_naerm_edges_rule_based_pj_p'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    
    critical_fac=['Compressor_Stations','Military_Bases']
    region='national'
    
    check_uncertainty(filedir,outdir,outfile,edgefile,region,K,p,m,critical_fac,iteration,case)
