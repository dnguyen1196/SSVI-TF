import numpy as np
from numpy.random import multivariate_normal
import warnings

# warnings.filterwarnings("error")
np.seterr(all="ignore")

"""
Probability functions, used for random sampling
as well as taking derivative of the various distributions
"""
epsilon = 1e-16

# def sigmoid(x):
#     try:
#         res = 1./(1 + np.exp(-x))
#         return np.minimum(np.maximum(x, epsilon), 1 - epsilon)
#     except Warning:
#         print("Overflow")
#         return np.minimum(np.maximum(x, epsilon), 1 - epsilon)

def sigmoid(x, epsilon=0):
    res = 1./(1 + np.exp(-x))
    #return res
    res = np.nan_to_num(res)
    #return np.minimum(np.maximum(res, epsilon), 1 - epsilon)i
    res = np.maximum(res, epsilon)
    return res

def sigmoid_scalar(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        if x < 0:
            return epsilon
        else:
            return 1

# sigmoid = np.vectorize(sigmoid_scalar)

# def poisson_link(f):
#     try:
#         res = np.log(1 + np.exp(f))
#         return np.maximum(res, epsilon)
#     except Warning:
#         print("warning")
#         return f

def poisson_link(f, epsilon=1e-16):
    res = np.log(1 + np.exp(f))
    res = np.nan_to_num(res)
    return np.maximum(res, epsilon)
    #return res

def poisson_link_scalar(f):
    try:
        res = np.log(1 + np.exp(f))
        return np.maximum(res, epsilon)
    except Warning:
        return np.maximum(res, epsilon)

# poisson_link = np.vectorize(poisson_link_scalar)
