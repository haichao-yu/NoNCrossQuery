import os
import time
import numpy as np
from scipy import sparse
import Precomputation
import CR


def run_cr_dblp(alpha=0.2, c=0.85, MaxIter=1000, epsilon=1e-15, q=121, s=0, d=19, k=10, dataset="../data/DBLP_NoN.npy"):
    """
    CrossRank evaluation on DBLP dataset
    
    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    :param MaxIter: the maximal number of iteration for updating ranking vector
    :param epsilon: a convergence parameter
    :param q: the ID of the query node interest (q = 121 by default, which represent the ID of Jiawei Han)
    :param s: the ID of the source domain-specific network (s = 0 by default, which represents the ID of KDD Conference)
    :param d: the ID of the target domain-specific network (d = 19 by default, which represents the ID of SIGMOD Conference)
    :param k: the number of retrieved nodes
    :return: top k author names
    """

    '''
    Load NoN data
    '''
    data = np.load(dataset).item()
    CoAuthorNets = data['CoAuthorNets']
    ConfNet = data['ConfNet']
    ConfDict = data['ConfDict']
    AuthorDict = data['AuthorDict']
    CoAuthorNetsID = data['CoAuthorNetsID']

    '''
    Rename networks
    '''
    G = sparse.csc_matrix(ConfNet)  # the main network
    g = CoAuthorNets.shape[1]
    A = CoAuthorNets  # the domain-specific networks
    A_ID = CoAuthorNetsID  # the IDs of nodes in domain-specific networks

    '''
    Precomputation, this step only needs to be done once for a dataset
    '''
    PrecompFileName = 'Precomp_Values_DBLP.npy'

    if os.path.isfile(PrecompFileName):
        print("A precomputation file has been detected ...")
    else:
        print("Precomputation starts ...")
        Precomputation.precomputation(A, A_ID, G, PrecompFileName)

    print("Load the precomputation file ...")
    data = np.load(PrecompFileName).item()
    I_n = data['I_n']
    Anorm = data['Anorm']
    Ynorm = data['Ynorm']

    '''
    Run CR
    '''
    # set initial scores
    start = time.time()

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

    # CR
    [r, Objs, Deltas] = CR.cr(Anorm, Ynorm, I_n, e, alpha, c, MaxIter, epsilon)

    # sort ranking scores in the target domain-specific network
    st = np.sum(DomainSizes[0, 0:d])
    ed = np.sum(DomainSizes[0, 0:d+1])
    rd_idx = np.arange(st, ed)
    rd = r[rd_idx]

    Sort_rd = np.flip(np.sort(rd.todense().getA1()), axis=0)
    Sort_Idx = np.flip(np.argsort(rd.todense().getA1()), axis=0)

    TopKResults = Sort_Idx[0:k]
    TopKResults = A_ID[0, d][TopKResults, 0]

    end = time.time()
    Runtime = end - start

    print("The running time of CR is " + str(Runtime) + " seconds.")

    TopKAuthorNames = AuthorDict[TopKResults - 1, 0]

    return TopKAuthorNames
