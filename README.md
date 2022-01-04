# Network_Sampling

This is the repo for the Network Sampling Comparison project which is advised by [Prof. Yichen Qin](https://business.uc.edu/faculty-and-research/departments/obais/faculty/yichen-qin.html) at University Of Cincinnati.

In this project, we try to answer the question of which sampling algorithm to use when targeting a certain type of graph statistic while the population network follows a specific network model.


Sampling networks is crucial to understanding the population networks, however it is difficult to determine which is the best performing sampling algorithms among a bunch of sampling algorithms. Depending on what is the type of the population network and what is the interested network statistics, best performing sampling algorithm varies a lot. In general, we have roughly split the sampling algorithms into four categories: Node sampling algorithms, Edge sampling algorithms, Traversal based sampling algorithms and Path based sampling algorithms. Each category has its own property and there are also many variations within each category. As a practitioner, it is quite hard to determine in advance which sampling algorithm may perform best in terms of certain type of network and network statistics. 

Here is an example of estimating global clustering coefficient where the population network follows from Erdős–Rényi (ER) model. It is shown that random node sampling (RN) and Metropolis–Hastings random walk sampling (MHRW) with induced subgraph perform well.
![27031641267068_ pic](https://user-images.githubusercontent.com/91963401/148006032-4ddb3c55-4664-4424-b8a0-5c43d6aba88f.jpg)
