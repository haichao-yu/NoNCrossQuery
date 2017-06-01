import math
import numpy as np
from scipy import sparse
import DijkstraExpansion


def extract_subnet(G, s, d, epsilon):
    """
    Extract a relevant subnetwork from the main network w.r.t. source and target domains
    
    :param G: the adjacency matrix of the main network
    :param s: the index of the source domain-specific network
    :param d: the index of the target domain-specific network
    :param epsilon: an error factor to control the accuracy of results
    :return: ID of domain networks
    """

    '''
    Initialization for expansion
    '''
    L_sd = float('inf')  # the shortest distance between s and d in G
    MaxRadius = (L_sd - math.log10(epsilon)) / 2.0  # the maximal radius to search
    Ns = np.array([])  # the neighborhoods of s
    Nd = np.array([])  # the neighborhoods of d
    rs = 0  # the radius of s
    rd = 0  # the radius of d
    SubG_Idx = np.array([], dtype=np.int64)  # the indices of nodes in the extracted main network
    Flag = 1  # the flag for the first overlap between neighborhoods of s and d

    '''
    Transform similarities in the main network to distances
    '''
    dG = G.sum(axis=1).getA()
    D_Gn = sparse.diags((dG ** (-0.5)).ravel())
    Gnorm = D_Gn.dot(G).dot(D_Gn)
    eps = np.spacing(1.0)
    Gnorm = np.maximum(Gnorm.todense().getA(), np.full(Gnorm.shape, eps))
    DisG = -np.log10(Gnorm)
    DisG = DisG - np.diag(np.diag(DisG))

    '''
    Heap initialization
    '''
    DisG = sparse.csr_matrix(DisG)  # csr format sparse matrix
    rp = DisG.indptr
    ci = DisG.indices
    vi = DisG.data
    n = len(rp) - 1  # the number of nodes

    Dis_s = np.full((n, 1), float('inf'))  # the distances from s to other nodes
    Hs = np.zeros((n, 1))  # the heap of node indices for s
    Ps = np.zeros((n, 1))  # the heap positions of nodes for s
    Len_s = 1  # the heap length of s
    Hs[Len_s - 1, 0] = s
    Ps[s, 0] = Len_s
    Dis_s[s, 0] = 0

    Dis_d = np.full((n, 1), float('inf'))  # the distances from d to other nodes
    Hd = np.zeros((n, 1))  # the heap of node indices for d
    Pd = np.zeros((n, 1))  # the heap positions of nodes for d
    Len_d = 1  # the heap length of d
    Hd[Len_d - 1, 0] = d
    Pd[d, 0] = Len_d
    Dis_d[d, 0] = 0

    '''
    Neighbourhood expansion loop
    '''
    while (rs <= MaxRadius and Len_s > 0) or (rd <= MaxRadius and Len_d > 0):

        if len(np.intersect1d(Ns, Nd)) == 1 and Flag == 1:  # the first overlap of two neighbourhoods

            Mid = int(np.intersect1d(Ns, Nd)[0])
            L_sd = Dis_s[Mid, 0] + Dis_d[Mid, 0]
            MaxRadius = (L_sd - math.log10(epsilon)) / 2.0
            Flag = 0

        if rs <= MaxRadius and Len_s > 0:

            [NextNgbr_s, Len_s, Hs, Ps, Dis_s] = DijkstraExpansion.dijkstra_expansion(rp, ci, vi, Hs, Ps, Dis_s, Len_s)
            Ns = np.hstack((Ns, NextNgbr_s))
            rs = Dis_s[NextNgbr_s, 0]

        if rd <= MaxRadius and Len_d > 0:

            [NextNgbr_d, Len_d, Hd, Pd, Dis_d] = DijkstraExpansion.dijkstra_expansion(rp, ci, vi, Hd, Pd, Dis_d, Len_d)
            Nd = np.hstack((Nd, NextNgbr_d))
            rd = Dis_d[NextNgbr_d, 0]

    '''
    Full relax
    
    When the above loop stops, the shortest distances between s and the nodes
    in Nd but not in Ns are not fully computed. This also applies to the shortest
    distances between d and the nodes in Ns but not in Nd. Thus, the following
    codes further relax these nodes.
    '''
    Ns_extra = np.setdiff1d(Nd, Ns)  # the nodes in Nd but not in Ns
    Nd_extra = np.setdiff1d(Ns, Nd)  # the nodes in Ns but not in Nd

    while Ns_extra.size > 0:  # Ns_extra is not empty

        [NextNgbr_s, Len_s, Hs, Ps, Dis_s] = DijkstraExpansion.dijkstra_expansion(rp, ci, vi, Hs, Ps, Dis_s, Len_s)
        if np.in1d(NextNgbr_s, Ns_extra):
            Ns = np.hstack((Ns, NextNgbr_s))
            Ns_extra = Ns_extra[Ns_extra != NextNgbr_s]  # remove the elem which equals to NextNgbr_s from Ns_extra

    while Nd_extra.size > 0:  # Nd_extra is not empty

        [NextNgbr_d, Len_d, Hd, Pd, Dis_d] = DijkstraExpansion.dijkstra_expansion(rp, ci, vi, Hd, Pd, Dis_d, Len_d)
        if np.in1d(NextNgbr_d, Nd_extra):
            Nd = np.hstack((Nd, NextNgbr_d))
            Nd_extra = Nd_extra[Nd_extra != NextNgbr_d]  # remove the elem which equals to NextNgbr_d from Nd_extra

    '''
    Further pruning Ns and Nd
    '''
    Nsd = np.hstack((Ns, Nd))
    Nsd = np.unique(Nsd)

    for i in range(len(Nsd)):

        u = int(Nsd[i])
        if Dis_s[u, 0] + Dis_d[u, 0] <= L_sd - math.log10(epsilon):
            SubG_Idx = np.hstack((SubG_Idx, u))

    return SubG_Idx
