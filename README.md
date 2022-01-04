# Network_Sampling

This is the repo for the Network Sampling Comparison project which is advised by [Prof. Yichen Qin](https://business.uc.edu/faculty-and-research/departments/obais/faculty/yichen-qin.html) at University Of Cincinnati.

In this project, we try to answer the question of which sampling algorithm to use when targeting a certain type of graph statistic while the population network follows a specific network model.

Here is an example of estimating global clustering coefficient where the population network follows from Erdős–Rényi model. It is shown that random node sampling (RN) and Metropolis–Hastings random walk sampling (MHRW) with induced subgraph perform well.
![27031641267068_ pic](https://user-images.githubusercontent.com/91963401/148006032-4ddb3c55-4664-4424-b8a0-5c43d6aba88f.jpg)


For distribution based network statistics, we give an example of estimating local clustering distribution where the population network follows from Barabási–Albert network. We present the performance of using snow ball sampling (SBS) with induced subgraph. As the sample sizes increase, the performances get better.
![27041641268163_ pic](https://user-images.githubusercontent.com/91963401/148007197-c8488b80-1f0f-4c81-9efb-f4d085eb33da.jpg)
