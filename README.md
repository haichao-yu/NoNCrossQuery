# NoNCrossQuery

This is an implementation of NoNCrossQuery (a special case of NoNCrossRank) algorithm with Python.

The goal of this algorithm is to solve:<br/>
**Given:** (1) an Network of Networks (NoN) R = <G, A, &theta;>, (2) a query node of interest from a source domain-specific network A<sub>s</sub>, (3) a target domain-specific network A<sub>d</sub>, and (4) an integer k;</br>
**Find:** the top-k most relevant nodes from the target domain-specific network A<sub>d</sub> w.r.t the query node.



## Functions

* **\_\_init\_\_.py:** program entry;
* **Precomputation.py:** CR and CQ precomputation (to obtain normalized A and normalized Y)
* **RunCQ_Basic.py:** run basic version of CrossQuery algorithm
* **RunCQ_Fast.py:** run fast version of CrossQuery algorithm
* **RunCQ_DBLP.py:** run CrossRank algorithm to solve CrossQuery problem
* **CQ_Basic.py:** CrossQuery-basic algorithm
* **CQ_Fast.py:** CrossQuery-fast algorithm
* **CR.py:** CrossRank algorithm
* **ExtractSubNet.py:** extract a relevant sub-network from the main network w.r.t. source and target domains, use the sub-network in the CrossQuery-fast algorithm
* **DijkstraExpansion.py:** conduct one step expansion of Dijkstra's algorithm


## Input/Output Format

### - Input

G: the adjacency matrix of main network<br/>
A: domain specific networks A = (A<sub>1</sub>, ..., A<sub>g</sub>)<br/>
&theta;: the one-to-one mapping function (mapping main node to domain-specific network)<br/>
s: the id of source domain-specific network<br/>
d: the id of target domain-specific network<br/>
e<sub>s</sub>: the query vector for source domain-specific network A<sub>s</sub> (i = 1, ..., g)<br/>
k: an integer indicating that top-k most relevant nodes from the target domain-specific network will be returned

### - Output
TopKAuthorNames: the top-k most relevant nodes from the target domain-specific network A<sub>d</sub> w.r.t the query node



## Reference
Ni, J., Tong, H., Fan, W., & Zhang, X. (2014, August). **Inside the atoms: ranking on a network of networks**. In Proceedings of the 20th *ACM SIGKDD* international conference on Knowledge discovery and data mining (pp. 1356-1365). ACM.