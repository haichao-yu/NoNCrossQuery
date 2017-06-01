import math


def dijkstra_expansion(rp, ci, vi, H, P, Dis, Len):
    """
    One step expansion of Dijkstra's algorithm
    
    :param rp: row pointer of csr matrix
    :param ci: column index of csr matrix
    :param vi: value index of csr matrix
    :param H: the heap of node indices
    :param P: the positions of nodes
    :param Dis: the distance vector of each node to the source/target node
    :param Len: the length of the leap
    :return
    """

    '''
    Pop the head off the heap
    '''
    u = int(H[0, 0])
    Tail = int(H[Len - 1, 0])
    H[0, 0] = Tail
    P[Tail, 0] = 1
    Len -= 1

    '''
    Maintain the min-heap
    '''
    [H, P] = MinHeap(H, P, Dis, Len)

    '''
    Relax the neighbors of u
    '''
    [Len, H, P, Dis] = Relax(rp, ci, vi, H, P, Dis, u, Len)

    return [u, Len, H, P, Dis]


def MinHeap(H, P, Dis, Len):
    """
    Maintaining min-heap
    """

    Pos = 1
    FirstVertex = int(H[Pos - 1, 0])

    '''
    Move the first node down the heap
    '''
    while True:

        Idx = 2 * Pos

        if Idx > Len:  # no child
            break
        elif Idx == Len:  # one child
            v = H[Idx - 1, 0]
        else:  # two children, pick the smaller one
            Left = int(H[Idx - 1, 0])
            Right = int(H[Idx, 0])
            v = Left
            if Dis[Right, 0] < Dis[Left, 0]:
                Idx += 1
                v = Right

        if Dis[int(FirstVertex), 0] < Dis[int(v), 0]:
            break
        else:
            H[Pos - 1, 0] = v
            P[int(v), 0] = Pos
            H[Idx - 1, 0] = FirstVertex
            P[FirstVertex, 0] = Idx
            Pos = Idx

    return [H, P]


def Relax(rp, ci, vi, H, P, Dis, u, Len):
    """
    Neighbor relax
    """

    for EdgeIdx in range(rp[u], rp[u + 1]):

        v = ci[EdgeIdx]  # v is a neighbor of u
        EdgeWeight = vi[EdgeIdx]

        # relax edge (u, v)
        if Dis[v, 0] > Dis[u, 0] + EdgeWeight:

            Dis[v, 0] = Dis[u, 0] + EdgeWeight
            Pos = P[v, 0]

            if Pos == 0:  # v is not in the heap

                Len += 1
                H[Len - 1, 0] = v
                P[v, 0] = Len
                Pos = Len

            # move v up the heap
            while Pos > 1:

                ParPos = int(math.floor(Pos / 2.0))
                Par = int(H[ParPos - 1, 0])

                if Dis[Par, 0] < Dis[v, 0]:
                    break
                else:
                    H[ParPos - 1, 0] = v
                    P[v, 0] = ParPos
                    H[int(Pos) - 1] = Par
                    P[Par] = Pos
                    Pos = ParPos

    return [Len, H, P, Dis]
