#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:20:33 2021

@author: anikat1
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


filedir='../output/naerm-EV/scalability/'
data_g_size=pd.read_csv(filedir+'scalability_G.csv')

X_G=data_g_size['num_node']+data_g_size['num_edge']
X_G=X_G.to_numpy()
Y_G=data_g_size['model_time'].to_numpy()

fig1 = plt.figure(figsize=(8,6))
ax = fig1.subplots()
line, = plt.plot(X_G, Y_G, color='b', marker='x',lw=2, label="DIHeN") 
m, b,c = np.polyfit(X_G, Y_G, 2)
line1,= plt.plot(X_G, m*X_G*X_G + b*X_G+c, color='r', lw=2, label="1.26x^2+(-1.8)x+111.2")
#ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=20)
ax.tick_params(axis='both', labelsize=20)
#ax.set_xticklabels(ax.get_xticks(), rotation=45)
#ax.set_yticklabels(ax.get_yticks(), fontsize=20,rotation=45)
ax.set_xlabel('Size of G (N+|F|)',fontsize=24)
ax.set_ylabel('Time (sec.)',fontsize=24)
plt.show()
print('line1:',m,b,c)
data_k_size=pd.read_csv(filedir+'scalability_k.csv')
X_K=data_k_size['k'].to_numpy()
print(X_K)
Y_K=data_k_size['model_time'].to_numpy()

fig2 = plt.figure(figsize=(8,6))
ax = fig2.subplots()
line, = plt.plot(X_K, Y_K, color='b', marker='x',lw=2, label="DIHeN") 
m, b = np.polyfit(X_K, Y_K, 1)
line2,= plt.plot(X_K, m*X_K + b, color='r', lw=2, label="4.3x+(-27.2)x")
ax.legend(fontsize=20)
ax.set_yticks([500,1000,1500,2000])
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel('k',fontsize=22)
ax.set_ylabel('Time (sec.)',fontsize=22)
plt.show()
print('line2:',m,b)



