import numpy as np
from scipy import sparse
import ExtractSubNet
import CQ_Basic


def cq_fast(Anorm, Y, G, q, s, d, k, alpha, c, epsilon, A_ID):
    """
    CrossQuery-Fast
    
    :param Anorm: the aggregated normalized adjacency matrix of domain-specific networks
    :param Y: the matrix encoding the cross-domain mapping information
    :param G: the adjacency matrix of the main network
    :param q: the ID of the query node of interest
    :param s: the ID of the source domain-specific network
    :param d: the ID of the target domain-specific network
    :param k: the number of retrieved nodes
    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    :param epsilon: an error factor to control the accuracy of results
    :param A_ID: the IDs of domain nodes in each domain-specific network
    :return: the ID of top k relevant authors from the target domain-specific network and the the ID of relevant domains
             of the source and target domains
    """

    '''
    Initialization
    '''
    vfunc = np.vectorize(lambda matrix: matrix.shape[0])  # define an element-wise operation
    DomainSizes = vfunc(A_ID)  # the number of domain nodes in each domain-specific network
    CumDomainSizes = np.cumsum(DomainSizes, 1)

    '''
    Extract relevant subnetwork from the main network
    '''
    SubG_Idx = ExtractSubNet.extract_subnet(G, s, d, epsilon)

    '''
    Calculate matrices
    '''
    g = len(SubG_Idx)
    SubA_Idx = np.array([], dtype=np.int64)

    for i in range(g):

        Hd = CumDomainSizes[0, int(SubG_Idx[i])] - DomainSizes[0, int(SubG_Idx[i])] + 1
        T1 = CumDomainSizes[0, int(SubG_Idx[i])]
        SubA_Idx = np.hstack((SubA_Idx, np.arange(Hd - 1, T1, dtype=np.int64)))

    DomainSizes = DomainSizes[0, SubG_Idx]
    A_ID = A_ID[0, SubG_Idx]
    A_ID = A_ID.reshape(1, len(A_ID))
    Anorm = Anorm.tocsc()
    Anorm = Anorm[SubA_Idx, :]
    Anorm = Anorm[:, SubA_Idx]
    s = np.nonzero(SubG_Idx == s)[0][0]
    d = np.nonzero(SubG_Idx == d)[0][0]

    SubG = G[SubG_Idx, :]
    SubG = SubG[:, SubG_Idx]
    dSubG = SubG.sum(axis=1)

    Dy = np.zeros((g, 1), dtype=object)  # cell matrix
    for i in range(g):
        Dy[i, 0] = dSubG[i, 0] * sparse.eye(DomainSizes[i])
    Dy = sparse.block_diag(Dy[:, 0])

    Y = Y[SubA_Idx, :]
    Y = Y[:, SubA_Idx]
    Dt = Dy - sparse.diags(Y.sum(axis=1).getA().ravel()).tocsc()
    Y = Y + Dt
    Dyn = Dy.power(-0.5)
    Ynorm = Dyn.dot(Y).dot(Dyn)

    tilde_c = (c + 2.0 * alpha) / (1.0 + 2.0 * alpha)
    W = (c / (c + 2.0 * alpha)) * Anorm + ((2.0 * alpha) / (c + 2.0 * alpha)) * Ynorm

    '''
    Apply CQ_Basic on the extracted NoN
    '''
    TopKResults = CQ_Basic.cq_basic(W, q, s, d, k, tilde_c, A_ID)

    return [TopKResults, SubG_Idx]
