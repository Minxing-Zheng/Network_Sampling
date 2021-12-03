#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import networkx as nx 
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 
import collections
import seaborn as sns
import math
import time
from scipy.stats import ks_2samp
import re
from queue import Queue
from collections import deque
from sklearn.metrics import mean_squared_error
import scipy.stats as ss
from scipy.sparse.linalg import eigs
import itertools
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from scipy import stats
from queue import LifoQueue
import prettytable as pt
import pickle
import operator
import sys
import community


# In[1]:


def RN(graph,sample_size=100):
    #random.seed(seed)
    nodes=random.sample(list(graph.nodes),sample_size,)   
    return nodes

def DN(graph,sample_size=100):
    nodes = [node for node in list(graph.nodes)]
    degrees = [float(graph.degree(node)) for node in list(graph.nodes)]
    degree_sum = sum(degrees)
    pro = [degree/degree_sum for degree in degrees]
    #np.random.seed(seed)
    sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False, p=pro)
    return sampled_nodes

def PN(graph,sample_size=100):
    nodes = [node for node in list(graph.nodes)]
    page_rank = nx.pagerank_scipy(graph)
    page_rank_sum = sum(page_rank.values())
    probabilities = [page_rank[node]/page_rank_sum for node in nodes]
    #np.random.seed(seed)
    sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False, p=probabilities)
    return sampled_nodes

# In[3]:


#a cost is defined as detecting a node's information
#we control the cost and determine which kind of subgraph we want
def RE(graph,sample_size=100):
    length=0
    sampled_edges=set()
    edges = [edge for edge in graph.edges()]
    #random.seed(seed)
    while length < sample_size:
        edge = random.choice(edges)
        sampled_edges.add(edge)
        length=len(set([y for x in sampled_edges for y in x]))
    return sampled_edges

#RandomNodeEdge Sampling
def RNE(graph,sample_size=100): 
    sampled_edges = set()
    length=0
    #random.seed(seed)
    while length < sample_size:
        source_node = random.choice(list(graph.nodes))
        try:
            target_node = random.choice([node for node in graph.neighbors(source_node)])
        except: continue
        edge = sorted([source_node, target_node])
        edge = tuple(edge)
        sampled_edges.add(edge)
        length=len(set([y for x in sampled_edges for y in x]))
    return sampled_edges

def HYB(graph,sample_size=100,p=0.8):
    sampled_edges = set()
    edges = [edge for edge in graph.edges()]
    length=0
    #random.seed(seed)
    while length < sample_size:
        score = random.uniform(0, 1)
        if score < p:
            source_node = random.choice(list(graph.nodes))
            try:
                target_node = random.choice([node for node in graph.neighbors(source_node)])
                edge = sorted([source_node, target_node])
                edge = tuple(edge)
            except:
                continue
        else:
            edge = tuple(random.choice(edges))
        sampled_edges.add(edge)
        length=len(set([y for x in sampled_edges for y in x]))
    return sampled_edges


# In[4]:


def SBS(graph,sample_size=100,k=3):#snowball sampling
    queue = Queue()
    #random.seed(seed);
    start_node = random.choice(list(graph.nodes))
    queue.put(start_node)
    
    nodes = set([start_node])
    t=0
    edges=set()
    #random.seed(seed)
    while len(nodes) < sample_size:
        try:
            source = queue.get()
            neighbors = [node for node in graph.neighbors(source)]
            unvisited= [node for node in neighbors if node not in nodes] 
            random.shuffle(unvisited)
            neighbors = unvisited[0:min(len(unvisited), k)]
            for neighbor in neighbors:
                nodes.add(neighbor)
                queue.put(neighbor)
                edges.add((source,neighbor))
                if len(nodes) >= sample_size:
                    break
            if queue.empty():
                raise IndexError
        except:raise IndexError
    return edges

def FFS_Geom(graph, sample_size=1000,p=3/7):#Forest Fire
    sampled_nodes = set()
    #random.seed(seed)
    start_node=random.choice(list(graph.nodes))
    node_queue = deque([start_node])
    edges=set()
    #np.random.seed(seed)
    while len(sampled_nodes) < sample_size:
        try:
            top_node = node_queue.popleft()
            sampled_nodes.add(top_node)
            neighbors = {neb for neb in graph.neighbors(top_node)}
            unvisited_neighbors = neighbors.difference(sampled_nodes)
            score = np.random.geometric(p)
            count = min(len(unvisited_neighbors), score)
            neighbors = random.sample(unvisited_neighbors, count)
            for neighbor in neighbors:
                if len(sampled_nodes) >= sample_size:
                    break
                sampled_nodes.add(neighbor)
                node_queue.extend([neighbor])
                edges.add((top_node,neighbor))
        except: raise IndexError
    return edges

def FFS_Prob(graph, sample_size=1000,p=1/2):#Forest Fire
    sampled_nodes = set()
    #random.seed(seed)
    start_node=random.choice(list(graph.nodes))
    node_queue = deque([start_node])
    edges=set()
    #np.random.seed(seed)
    while len(sampled_nodes) < sample_size:
        try:
            top_node = node_queue.popleft()
            sampled_nodes.add(top_node)
            neighbors = {neb for neb in graph.neighbors(top_node)}
            unvisited_neighbors = neighbors.difference(sampled_nodes)
            score = int(p*len(neighbors))
            count = min(len(unvisited_neighbors), score,1)
            neighbors = random.sample(unvisited_neighbors, count)
            for neighbor in neighbors:
                if len(sampled_nodes) >= sample_size:
                    break
                sampled_nodes.add(neighbor)
                node_queue.extend([neighbor])
                edges.add((top_node,neighbor))
        except: raise IndexError
    return edges


def BFS(graph,sample_size=100,):
    number_of_nodes=sample_size
    queue = Queue()
    #random.seed(seed)
    start_node = random.choice(list(graph.nodes))
    queue.put(start_node)
    nodes = set([start_node])
    edges=set()
    while len(nodes) < number_of_nodes:
        try:
            source = queue.get()
            neighbors = [node for node in graph.neighbors(source)]
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in nodes:
                    nodes.add(neighbor)
                    edges.add((source, neighbor))
                    queue.put(neighbor)
                    if len(nodes) >= number_of_nodes:
                        break
            if queue.empty():
                raise IndexError
        except:raise IndexError
    return edges


def DFS(graph,sample_size=100):
    number_of_nodes=sample_size
    queue = LifoQueue()
    #random.seed(seed)
    start_node = random.choice(list(graph.nodes))
    queue.put(start_node)
    nodes = set([start_node])
    edges=set()
    while len(nodes) < number_of_nodes:
        try:
            source = queue.get()
            neighbors = [node for node in graph.neighbors(source)]
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in nodes:
                    nodes.add(neighbor)
                    edges.add((source, neighbor))
                    queue.put(neighbor)
                    if len(nodes) >= number_of_nodes:
                        break
            if queue.empty():
                raise IndexError
        except:raise IndexError
    return edges


# In[5]:


def RW(graph,sample_size=100):
    #random.seed(seed)
    current_node = random.choice(list(graph.nodes))
    sampled_nodes = set([current_node])
    edges=set()
    
    if len(nx.node_connected_component(graph,current_node))<sample_size:
        raise IndexError
    
    while len(sampled_nodes) < sample_size:
        try:
            neighbors = graph.neighbors(current_node)
            neighbor = random.choice([neighbor for neighbor in neighbors])
            sampled_nodes.add(neighbor)
            edges.add((current_node,neighbor))
            current_node=neighbor  
        except: raise IndexError
    return edges

def MHRW(graph,sample_size=100):
    #random.seed(seed)
    current_node = random.choice(list(graph.nodes))
    sampled_nodes = set([current_node])
    edges=set()
    if len(nx.node_connected_component(graph,current_node))<sample_size:
        raise IndexError
        
    while len(sampled_nodes) <sample_size:
        try:
            score = random.uniform(0, 1)
            neighbors = graph.neighbors(current_node)
            new_node = random.choice([neighbor for neighbor in neighbors])
            ratio = float(graph.degree(current_node))/float(graph.degree(new_node))
            if score < ratio:
                edges.add((current_node,new_node))
                current_node = new_node
                sampled_nodes.add(current_node)       
        except:raise IndexError
    return edges

def FS(graph,sample_size=100,number_of_seeds=10):
    number_of_nodes=sample_size
    
    nodes = [node for node in list(graph.nodes)]
    #random.seed(seed)
    seeds = random.sample(nodes, number_of_seeds)
    
    nodes=set()
    edges=set()
    #np.random.seed(5)
    while len(nodes) < number_of_nodes:
        try:
            #reweight
            seed_weights = [graph.degree(seed) for seed in seeds]
            weight_sum = np.sum(seed_weights)
            seed_weights = [float(weight)/weight_sum for weight in seed_weights]

            #sample
            sample = np.random.choice(seeds, 1, replace=False, p=seed_weights)[0]
            index = seeds.index(sample)
            new_seed = random.choice([neb for neb in graph.neighbors(sample)])
            edges.add((sample, new_seed))
            nodes.add(sample);nodes.add(new_seed)
            seeds[index] = new_seed
        except:raise IndexError
        
    return edges


# In[18]:


def SP(graph,sample_size=100):
    number_of_nodes=sample_size
    nodes=set()
    #random.seed(seed)
    while len(nodes) < number_of_nodes:
        source=random.choice(list(graph.nodes))
        target =random.choice(list(graph.nodes))
        if source != target:
            try:
                path = nx.shortest_path(graph, source, target)
                for node in path:
                    nodes.add(node)
                    if len(nodes) >= number_of_nodes:
                        break
            except:continue
    return nodes


# In[4]:


def get_list(graph,func,sample_size=100,iteration=100):
    SG_list=set()
    times=0
    if func.__name__ in ['RE','RNE','HYB','SBS','FFS_Geom','BFS','DFS','RW','MHRW','FS']:
        # return sampled edges
        for i in range(0,iteration+1000000):
            try:
                sample=func(graph,sample_size)  
                SG=graph.edge_subgraph(sample)
                SG_list.add(SG)
                times+=1
            except:continue
            if times==iteration:break
                
    else:#return sampled nodes
        for i in range(0,iteration+1000000):
            try:
                sample=func(graph,sample_size)  
                SG=graph.subgraph(sample)
                SG_list.add(SG)
                times+=1
            except:continue
            if times==iteration:break
    if len(SG_list)<iteration:
        print(func.__name__+'not sample enough')
    return list(SG_list)


# In[ ]:


def Degree_Related(SG_List,graph,Degree,density_G):
    Range=range(min(Degree),max(Degree)+1)
   

    Rate_G=[]
    for i in Range:
        rate_g=len([y for y in Degree if y<=i])/len(Degree)
        Rate_G.append(rate_g)
        
    Rate_Mse=[];AveD_SG=[]
    for SG in SG_List:
        Degree_SG=[]
        for node in list(SG.nodes):
            Degree_SG.append(graph.degree(node))
        AveD_SG.append(np.mean(Degree_SG))

        Rate_SG=[]
        for i in Range:
            rate_sg=len([y for y in Degree_SG if y<=i])/len(Degree_SG);
            Rate_SG.append(rate_sg)
        Rate_Mse.append(Rate_SG)

    SG_density=[]
    for SG in SG_List:
        density=nx.density(SG)
        SG_density.append(density)
        
    #return [[nmse,bias,var],l2_mse,[nmse_density,bias_density,var_density],[SG_density,density_G],[AveD_SG,np.mean(Degree)],[Rate_Mse,Rate_G]]
    return ([AveD_SG,Degree],[Rate_Mse,Rate_G],[SG_density,density_G])

def ClustCoeff_Related(SG_List,graph,GCC,LCC):
    #GCC
    SG_gcc=[]
    for SG in SG_List:
        GCC_SG=nx.transitivity(SG)
        SG_gcc.append(GCC_SG)

    LCC_SG=[];Ave_LCC_SG=[];L2=[]
    Range=np.arange(0,1+0.0005,0.0005)
    
    Rate_G=[]
    for i in Range:
        rate=len([y for y in LCC if y<=i])/len(LCC)
        Rate_G.append(rate)#real       
    
    Rate_Mse=[];LCC_SG=[]
    for SG in SG_List:
        lcc_sg=list(nx.clustering(SG).values())
        ave_lcc=np.mean(lcc_sg)
        Ave_LCC_SG.append(ave_lcc)

        Rate_SG=[]
        for i in Range:
            rate=len([y for y in lcc_sg if y<=i])/len(lcc_sg)
            Rate_SG.append(rate)#sample
        Rate_Mse.append(Rate_SG)
        
    return ([[SG_gcc,GCC],[Ave_LCC_SG,LCC],[Rate_Mse,Rate_G]])

def Community_Related(SG_List,mod_model,num_mod):
    Num_mod_SG=[];Mod_SG=[]
    for SG in SG_List:
        try:
            part_sg = community.best_partition(SG)
            mod_sg = community.modularity(part_sg,SG)
            num_mod_sg=len(np.unique(list(part_sg.values())))
        except:
            mod_sg=0;num_mod_sg=0
            
        Num_mod_SG.append(num_mod_sg)
        Mod_SG.append(mod_sg) 
    
    #return([nmse_mod,bias_mod,var_mod],[nmse_modnum,bias_modnum,var_modnum],[Num_mod_SG,Mod_SG])
    return ([Num_mod_SG,num_mod],[Mod_SG,mod_model])

# In[ ]:


def X_Hat_List_func(SG_List):
    
    
    X_hat_SG=[]
    for SG in SG_List:
        A=nx.adjacency_matrix(SG)
        eigval,eigvec=np.linalg.eig(A.A)
        idx = eigval.argsort()[::-1]   
        eigval_first_sd =eigval[idx][0:5]
        eigvec_first_sd = eigvec[:,idx][:,0:5]
        S=np.diag(eigval_first_sd)
        S_sqrt=np.sqrt(S)
        X_hat=eigvec_first_sd@S_sqrt
        X_hat_SG.append(X_hat)
    
    return X_hat_SG

def func_largest(SG_List,eigval_G):
    num_spec_SG=[];Eig_max_sg=[]
    Eig_max_sg_2=[];Eig_max_sg_3=[]
    for SG in SG_List:
        A=nx.adjacency_matrix(SG)
        n=A.shape[0]
        eigval,eigvec=np.linalg.eig(A.A)
        idx = eigval.argsort()[::-1]   
        #eigvec_first_sd = eigvec[:,idx][:,0:5]

        Eig_max_sg.append(float(eigval[idx][0])/n)
        Eig_max_sg_2.append(float(eigval[idx][1])/n)
        Eig_max_sg_3.append(float(eigval[idx][2])/n)

    return([Eig_max_sg,Eig_max_sg_2,Eig_max_sg_3,eigval_G])


# In[2]:

#ER_model_high.pkl
model_type=str(sys.argv[1])
density_type=str(sys.argv[2])
stat_type=str(sys.argv[3])
set_seed=int(sys.argv[4])
with open(f"model_save/{model_type}_model_{density_type}.pkl",'rb') as f:
    model=pickle.load(f)


if stat_type=='degree':
    #degree related
    model_degree=[d for n,d in model.degree()]
    model_avedegree=np.mean(model_degree)
    model_density=nx.density(model)
    print("degree finished")

if stat_type=='clustering':
    #clustering related
    model_gcc=nx.transitivity(model) 
    model_lcc=list(nx.clustering(model).values())
    model_avelcc=np.mean(list(nx.clustering(model).values()))
    print("clustering finished")


if stat_type=='community':
    part_model = community.best_partition(model)
    mod_model = community.modularity(part_model,model)
    clustnum_model=len(np.unique(list(part_model.values())))

if stat_type=='eigenvalue':
    A=nx.adjacency_matrix(model)
    eigval,eigvec=np.linalg.eig(A.A)
    eigval=eigval.astype(np.float)
    idx = eigval.argsort()[::-1]   
    eigval =eigval[idx]
    #eigvec = eigvec[:,idx]


Name=['RN','DN','PN','RE','RNE','HYB','SBS','FFS_Geom','BFS','DFS','RW','MHRW','FS','SP']
#size_scale=[5]
size_scale=[1000]
size_ratio=[x/model.number_of_nodes() for x in size_scale]
size_ratio=["{:.0%}".format(x) for x in size_ratio]


# In[ ]:
num_iter=100
SG_List=[]
for size in size_scale:
    print(size)
    Name_List=[]
    for name in Name:
        Name_List.append(name)
        List=get_list(model,eval(name),sample_size=size,iteration=num_iter);SG_List.append(List)
        LList=[]
        if name in ['RE','RNE','HYB','SBS','FFS_Geom','BFS','DFS','RW','MHRW','FS']:
            for SG in List:
                nodes=[y for x in list(SG.edges) for y in x]
                LList.append(model.subgraph(nodes))
            List=LList
            Name_List.append('Induced_'+name);SG_List.append(List)        
print('success get list')

if stat_type=='degree':
    List=[]
    for x in range(len(size_ratio)):
        Degree=[]
        for y in SG_List[x*len(Name_List):(x+1)*len(Name_List)]:
            degree=Degree_Related(SG_List=y,graph=model,Degree=model_degree,density_G=model_density)
            Degree.append(degree) 
        List.append(Degree)

if stat_type=='clustering':
    List=[]
    for x in range(len(size_ratio)):#['1%', '5%', '10%', '20%']
        ClustCoeff=[]
        for y in SG_List[x*len(Name_List):(x+1)*len(Name_List)]:
            cc=ClustCoeff_Related(SG_List=y,graph=model,GCC=model_gcc,LCC=model_lcc)
            ClustCoeff.append(cc)
        List.append(ClustCoeff)

if stat_type=='community':
    List=[]
    for x in range(len(size_ratio)):#['1%', '5%', '10%', '20%']
    #for x in range(1):
        Community=[]
        for y in SG_List[x*len(Name_List):(x+1)*len(Name_List)]:
            com=Community_Related(SG_List=y,mod_model=mod_model,num_mod=clustnum_model);Community.append(com)
        List.append(Community)

if stat_type=="eigenvalue":
    List=[]
    for x in range(len(size_ratio)):
        l=[]
        for y in SG_List[x*len(Name_List):(x+1)*len(Name_List)]:
            result=func_largest(SG_List=y,eigval_G=eigval)
            l.append(result)
        List.append(l)


with open(f"{model_type}_{density_type}_{stat_type}_{set_seed}.pkl", 'wb') as f: 
        pickle.dump(List, f)


