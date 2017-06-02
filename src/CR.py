import numpy as np
from scipy.sparse.linalg import norm


def cr(Anorm, Ynorm, I_n, e, alpha, c, MaxIter, epsilon):
    """
    Cross Rank

    :param Anorm: the aggregated normalized adjacency matrix of domain-specific networks
    :param Ynorm: the normalized matrix encoding the cross-domain mapping information
    :param I_n: an identity matrix of size n x n
    :param e: the query vector
    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    :param MaxIter: the maximal number of iteration for updating raking vector
    :param epsilon: a convergence parameter

    :returns r: the ranking vector
    :returns Objs: objective values
    :returns Deltas: difference between objective values
    """

    '''
    Initialization
    '''
    # initialize ranking vector
    r = e.astype(np.float64)

    # initialize parameters
    gamma = c / (1.0 + 2.0 * alpha)
    kappa = 2.0 * alpha / (1.0 + 2.0 * alpha)
    eta = (1.0 - c) / (1.0 + 2 * alpha)

    # convergence analysis parameters:
    # either the difference between objective values or ranking vector norms can be used as a measure of convergence

    # J1 = J_CR.j_cr(Anorm, Ynorm, I_n, r, e, alpha, c)  # objective value measure
    J1 = r  # ranking vector norm measure
    J1 = (np.around(J1 * 1e16)) / 1e16
    delta = 99999
    Objs = []
    Deltas = []
    Iter = 1

    '''
    Power method update loop
    '''
    while delta > epsilon and Iter <= MaxIter:
        # update r
        M = gamma * Anorm + kappa * Ynorm
        r = M.dot(r) + eta * e

        # convergence analysis
        J2 = J1
        # J1 = J_CR.j_cr(Anorm, Ynorm, I_n, r, e, alpha, c)  # objective value measure
        J1 = r  # ranking vector norm measure
        J1 = (np.around(J1 * 1e16)) / 1e16
        # delta = J2 - J1  # objective value measure
        delta = norm(J2 - J1, ord=1)  # ranking vector norm measure

        # Objs = np.hstack((Objs, J1))  # objective value measure
        Objs = np.hstack((Objs, norm(J1, ord=1)))  # ranking vector norm measure
        Deltas = np.hstack((Deltas, delta))

        Iter += 1

    return [r, Objs, Deltas]
