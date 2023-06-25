import numpy as np
from scipy.linalg import norm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
# import shapley_residuals_llm.util as util # for testing
import util
import pickle

# Game helper functions -----


def isPowerOfTwo(n):
    return (n != 0) and ((n & (n-1))== 0)

def edge(a, b):
    # there is an edge from a to b in the oriented incidence matrix on the hypercube
    # if a XOR b is a power of 2
    return isPowerOfTwo(a ^ b)

def compute_d(n, i = -1):
    # actually compute the operator
    # this is done naively and slow, so we hide it behind memoization in get_d
    # TODO: Just get all 2^n permutations such that sum = 1
    dim = 2**n
    cols = dim
    rows = 2**(n-1) * n
    D = lil_matrix((rows, cols))
    cur_row = 0
    for f in range(dim):
        for t in range(f+1,dim):
            if edge(f,t):
                index = int(np.log2(abs(f-t)))
                if (i == -1) or (index == i):
                    D[cur_row,f] = -1
                    D[cur_row,t] = 1
                cur_row += 1
    return D.tocsr()
    
d_cache = {}
def get_d(n, i = -1):
    """get_d(n, i = -1) 
        returns the derivative operator for F players. 
        If i != -1, returns d_v for player v, by Stern's definition of d_i"""
    global d_cache
    if (n, i) in d_cache:
        return d_cache[(n, i)]

    fname = "__dcache__/%s-%s.pcl" % (n, i)
    try:
        f = open(fname, "rb")
        d_cache[(n, i)] = pickle.load(f)
    except FileNotFoundError:
        d_cache[(n, i)] = compute_d(n, i)
        with open(fname, "wb") as fout:
            pickle.dump(d_cache[(n, i)], fout)

    return d_cache[(n, i)]

def getShapleyProjection(v):
    n = int(np.log2(len(v)))
    results = []
    residualGame = []
    D = get_d(n)
    Divs = []
    for i in range(n):
        Di = get_d(n, i)
        Div = Di.dot(v)
        lsqrt_result = lsqr(D, Div)
        vi = lsqrt_result[0]
        results.append(vi - vi[0])
        residualGame.append(D.dot(vi) - Div)

    results = np.array(results).T

    residualGame = np.array(residualGame).T

    # sanity checks
    if norm(results.sum(axis=1) - (v - v[0])) > 1e-4:
        print("SANITY CHECK FAILED")
        print("Norm of difference between sum vi and v: %s" % norm(results.sum(axis=1) - (v - v[0])))
    if norm(residualGame.sum(axis=1)) > 1e-4:
        print("SANITY CHECK FAILED")
        print("Norm of sum of residuals: %s" % norm(residualGame.sum(axis=1)))
    
    return results, residualGame, D.dot(v)

# def test_get_d():
#     for i in range(2,10):
#         d = util.get_d(i)
#         ds = get_d(i)
#         if (np.array(d - ds.todense()) ** 2).sum() > 0.0:
#             raise Exception("Test failed at i=%s" % i)
#         for j in range(i):
#             d = util.get_d(i, j)
#             ds = get_d(i, j)
#             if (np.array(d - ds.todense()) ** 2).sum() > 0.0:
#                 raise Exception("Test failed at i=%s, j=%s" % (i, j))
#     print("ok")


# def getShapleyResiduals(v):
#     results, residualGame, origGame = getShapleyProjection(v)
#     return np.flip(results[-1]), norm(residualGame)/norm(origGame)

# def getShapleyPartialResiduals(v):
#     results, residualGame, origGame = getShapleyProjection(v)
#     return np.flip(results[-1]), np.flip(norm(residualGame, axis = 0)/norm(origGame, axis = 0))

