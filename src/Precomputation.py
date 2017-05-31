import numpy as np
from scipy import sparse


def precomputation(A, A_ID, G, PrecompFileName):
    """
    CR and CQ precomputation
    :param A: the domain-specific networks
    :param A_ID: the corresponding IDs of domain-specific networks in A
    :param G: the adjacency matrix of the main network
    :param PrecompFileName: the file name to store precomputation results
    """

    '''
    Initialization
    '''
    g = A.shape[1]  # the number of domain-specific networks

    vfunc = np.vectorize(lambda matrix: matrix.shape[0])  # define an element-wise operation
    ns = vfunc(A_ID)  # the number of domain nodes in each domain-specific network

    n = ns.sum()  # the total number of domain nodes (including common nodes)
    I_n = sparse.eye(n)  # n x n sparse identity matrix

    '''
    Normalize A: Anorms[0, i] = D^(-0.5) dot A dot D^(-0.5), where D is the degree matrix of A[0, i]
    '''
    Anorms = np.zeros(A.shape, dtype=object)  # create a block matrix (cell matrix in MatLab) with A's shape

    for i in range(g):

        D = A[0, i].sum(axis=1)  # sum matrix A[0, i] over axis=1 (now D is the type of numpy.matrix)
        D = D.getA()  # convert D to the type of numpy.ndarray
        D = D ** (-0.5)  # performs element-wise power
        D = D.ravel()  # get a flattened array
        D = sparse.diags(D)  # construct a diagonal matrix which is sparse

        Anorms[0, i] = (D.dot(A[0, i]).dot(D))

    # create a block diagonal matrix from provided matrice
    Anorm = sparse.block_diag(Anorms[0, :])

    '''
    Common node mapping (get the block matrix O)
    '''
    Dy = np.zeros((g, 1), dtype=object)  # degree matrix of Y
    dG = G.sum(axis=1)  # degree of main nodes

    # compute cumulative ns to reduce redundant computation
    cumulative_ns = np.cumsum(ns[0, :])
    cumulative_ns = np.hstack(([0], cumulative_ns))

    # initialize row, col, data for constructing coo_matrix (A[row[k], col[k]] = data[k])
    row = np.array([], dtype=np.int64)
    col = np.array([], dtype=np.int64)
    data = np.array([], dtype=np.float64)

    for i in range(g):

        Dy[i, 0] = dG[i, 0] * sparse.eye(ns[0, i], dtype=np.float64)

        for j in range(i, g):

            proj = np.intersect1d(A_ID[0, i], A_ID[0, j])  # common elements of A_ID[0, i] and A_ID[0, j]
            I1 = np.in1d(A_ID[0, i], proj).nonzero()[0]  # indices of common elements in A_ID[0, i]
            I2 = np.in1d(A_ID[0, j], proj).nonzero()[0]  # indices of common elements in A_ID[0, j]
            Oij = sparse.coo_matrix((np.ones(len(proj), dtype=np.float64), (I1, I2)), shape=(ns[0, i], ns[0, j]))

            O_ij_block = Oij.multiply(G[i, j])  # (i, j)th block of O
            O_ij_block.eliminate_zeros()  # in-place operation!!!
            O_ij_block = O_ij_block.tocoo()

            row = np.hstack((row, O_ij_block.row + cumulative_ns[i]))
            col = np.hstack((col, O_ij_block.col + cumulative_ns[j]))
            data = np.hstack((data, O_ij_block.data))

    O = sparse.coo_matrix((data, (row, col)), shape=(n, n))

    O = sparse.triu(O) + sparse.triu(O).transpose() - sparse.diags(O.diagonal())

    '''
    Construct normalized Y
    '''
    Dy = sparse.block_diag(Dy[:, 0])
    Dy = Dy.tocsc()
    Do = sparse.diags(O.sum(axis=1).getA().ravel())
    Do = Do.tocsc()
    # print Dy[0, 0]
    # print Do[0, 0]
    # print Dy[0, 0] - Do[0, 0]
    Dt = Dy - Do  # have precision problem, cause small entries in sparse matrix
    Y = O + Dt
    Dyn = Dy.power(-0.5)
    Ynorm = Dyn.dot(Y).dot(Dyn)

    # eliminate small entries caused by precision problem (optional)
    # number of nonzero entries (nnz): 3463488 -> 3332249
    Ynorm_data = Ynorm.data
    vfunc = np.vectorize(lambda elem: elem if elem >= 1e-15 else 0.0)
    Ynorm_data = vfunc(Ynorm_data)
    Ynorm = sparse.csc_matrix((Ynorm_data, Ynorm.indices, Ynorm.indptr), shape=Ynorm.shape)
    Ynorm.eliminate_zeros()

    '''
    Save precomputed matrices
    '''
    data = {}
    data['Anorm'] = Anorm
    data['Ynorm'] = Ynorm
    data['I_n'] = I_n
    np.save(PrecompFileName, data)
