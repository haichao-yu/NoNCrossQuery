import numpy as np
from scipy import sparse


def cq_basic(W, q, s, d, k, tilde_c, A_ID):
    """
    CrossQuery-Basic
    
    :param W: the transition matrix
    :param q: the ID of the query node of interest
    :param s: the ID of the source domain-specific network
    :param d: the ID of the target domain-specific network
    :param k: the number of retrieved nodes
    :param tilde_c: the normalized parameter for query preference
    :param A_ID: the IDs of domain nodes in each domain-specific network
    :return: the IDs of top k relevant authors from the target domain-specific network
    """

    '''
    Initialization
    '''
    g = A_ID.shape[1]  # the number of domain-specific networks

    vfunc = np.vectorize(lambda matrix: matrix.shape[0])  # define an element-wise operation
    DomainSizes = vfunc(A_ID)  # the number of domain nodes in each domain-specific network

    e = np.array([])  # initialize query vector

    for i in range(g):

        tmp_e = np.zeros((DomainSizes[0, i],))

        if i == s:
            tmp_e[A_ID[0, i].ravel() == q] = 1  # use g to replace 1 if NoN size is large

        e = np.hstack((e, tmp_e))

    e = e.reshape(len(e), 1)
    e = sparse.csc_matrix(e)

    Wmax = W.max(axis=1)
    Wmax = Wmax.todense().A

    st = np.sum(DomainSizes[0, 0:d])
    ed = np.sum(DomainSizes[0, 0:d+1])
    S = np.arange(st, ed)

    Iter = 1  # iteration number
    p = e  # the random walk vector
    Lower = (1.0 - tilde_c) * p[S, 0]  # the lower bound vector

    '''
    Score upper and lower bounds update loop
    '''
    while len(S) > k:

        '''
        Update random walk scores.
        
        Since MATLAB (here is SciPy) sparse matrix multiplication time complexity is proportional to the number
        of non-zero entries, it is more efficient in practice to directly multiply W and p than
        applying BFS search before this multiplication in each iteration.
        
        One can apply BFS one layer search in each iteration by using the function bfs_layer()
        '''
        p = W.dot(p)

        # update upper and lower bounds
        Lower = Lower + (1 - tilde_c) * (tilde_c ** Iter) * p[S, 0]
        Upper = Lower + (tilde_c ** (Iter + 1)) * Wmax[S]

        # update threshold by the kth lower bound (must have problem in here)
        kthSmallestValue = np.partition(-Lower.todense().A1, k - 1)[k - 1]  # zero-based index here
        Theta = -kthSmallestValue

        flattenUpper = np.ravel(Upper)
        S = S[flattenUpper >= Theta]
        Lower = Lower[flattenUpper >= Theta, 0]
        Upper = Upper[flattenUpper >= Theta, 0]

        # avoid duplicates
        if np.amax((Upper - Lower).A) < 1e-15:
            flattenLower = Lower.todense().A1
            SelectIdx = np.nonzero(flattenLower > Theta)[0]
            Duplicates = np.nonzero(flattenLower == Theta)[0]
            SelectIdx = np.hstack((SelectIdx, Duplicates))
            SelectIdx = SelectIdx[0:k]
            S = S[SelectIdx]

        Iter += 1

    TopKResults = S
    TopKResults = TopKResults - np.sum(DomainSizes[0, 0:d])
    TopKResults = A_ID[0, d][TopKResults, 0]

    return TopKResults
