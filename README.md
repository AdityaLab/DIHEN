# urbannet-powerflow

`DATA:`

1. domain-based national: data/naerm/
2. distance-based national: data/urbannet/
3. regional: data/urbannet/
4. edge files to evaluate criticality criteria: 
	1. data/v9/
	2. data/naerm-edge-files/

`NOTE: some files (national_transmission_lines.csv, EIC_transmission_lines.csv) exceeds git limit hence not uploaded see this in google drive directory ornl/powerflow-dihen/
`

`Code for distance-based network:` 
path: distance-based-scripts/

1. ablation_model.py  get results for ablation models
e.g.: python3 ablation_model.py data/urbannet/ result_ablation_national national

`Input parameters:`
a. data_path: data/urbannet/
b. outputfilename: result_ablation_national
c. region: national/regional
	1. national
	2. TX
	3. EIC

2. baseline_model.py: get results for baseline 
e.g.: python3 vary_k_urbannet.py ../data/urbannet/ result_baselines 50

`Input parameters:`
a. data_path: ../data/urbannet/
b. outputfilename: result_baselines
c. k: user defined int value, for paper we use 50

3. vary_k_urbannet.py: get results varying size k for national and regional network
e.g.: python3 vary_k_urbannet.py ../data/urbannet/ vary_k national

`Input parameters:`
a. data path: ../data/urbannet/
b. output file name: vary_k
c. region: network name
	1. national
	2. TX
	3. EIC
4. uncertainty_bar_urbannet.py: get results to compute error bar for every ablation model on distance-based national

e.g.: python3 ../data/urbannet/ uncertain_national national 50 1 

`Input parameters:`

a. data_path: ../data/urbannet/
b. output file name: uncertain_national
c. region: network_name
	1. national
	2. TX
	3. EIC
d. k: user defined int value for size k, for paper we use 50
e. case: int value between 1-5, where,
 1-> DIHEN
 2-> CB (U)
 3-> FP (K)
 4-> NER (Sibling-dist)
 5-> Random_Gaussian

5. baseline_spread_k.py: results to obtain number of spread for baseline models vary size k

e.g., python3 ../data/urbannet/ spread_vary_k national

`Input parameters:`

a. data path: ../data/urbannet/
b. output file name: spread_vary_k
c. region name:
	1. national
	2. TX
	3. EIC

6. critical_rank_full_urbannet.py : rank all nodes in the network considering k= #nodes in the network

`Code for domain-based network:`

path: domain-based-scripts/

1. naerm_ablation_model.py : get ablation results
2. naerm_baseline_model.py : get baseline results
3. naerm_vary_k.py : get results varying size k
4. uncertainty_bar_naerm.py : get results to compute error bar for every ablation model
e.g.: python3 ../data/naerm/ 500 1 

`Input parameters:`

a. data_path: ../data/naerm/
b. k: 500
c. case: int value between 1-5, where,
 1-> DIHEN
 2-> CB (U)
 3-> FP (K)
 4-> NER (Sibling-dist)
 5-> Random_Gaussian

5. critical_rank_full_naerm.py : rank all nodes in the network considering k= #nodes in the network

`Code for plotting:`

path: viz-scripts/ 
