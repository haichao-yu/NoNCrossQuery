import os
import time
import numpy as np
from scipy import sparse
import Precomputation
import CQ_Basic


# note that python index start from 0, while matlab index start from 1.
def run_cq_basic(alpha=0.2, c=0.85, q=121, s=0, d=19, k=10):
    """
    CrossQuery-Basic evaluation on DBLP dataset
    
    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    :param q: the ID of the query node of interest (q = 121 by default, which represent the ID of Jiawei Han)
    :param s: the ID of the source domain-specific network (s = 0 by default, which represents the ID of KDD Conference)
    :param d: the ID of the target domain-specific network (d = 19 by default, which represents the ID of SIGMOD Conference)
    :param k: the number of retrieved nodes
    :return: the names of top k relevant authors from the target domain-specific network
    
    Looking at ConfDict in ../data/DBLP_NoN.npy to determine source and target domain IDs, s and d.
    
    Looking at AuthorDict in ../data/DBLP_NoN.npy to determine the ID of the query node q.
    """

    '''
    Load NoN data
    '''
    data = np.load("../data/DBLP_NoN.npy").item()
    CoAuthorNets = data['CoAuthorNets']
    ConfNet = data['ConfNet']
    ConfDict = data['ConfDict']
    AuthorDict = data['AuthorDict']
    CoAuthorNetsID = data['CoAuthorNetsID']

    '''
    Rename networks
    '''
    G = sparse.csc_matrix(ConfNet)  # the main network
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
    Run CQ_Basic
    '''
    # initialization
    start = time.time()
    tilde_c = (c + 2.0 * alpha) / (1.0 + 2.0 * alpha)
    W = (c / (c + 2.0 * alpha)) * Anorm + ((2.0 * alpha) / (c + 2.0 * alpha)) * Ynorm

    # CQ_Basic
    TopKResults = CQ_Basic.cq_basic(W, q, s, d, k, tilde_c, A_ID)
    end = time.time()
    Runtime = end - start

    print("The running time of CQ_Basic is " + str(Runtime) + " seconds.")

    TopKAuthorNames = AuthorDict[TopKResults - 1]

    return TopKAuthorNames
