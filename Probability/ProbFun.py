import numpy as np
from numpy.random import multivariate_normal

"""
Probability functions, used for random sampling
as well as taking derivative of the various distributions
"""

"""
Normal distribution
"""
def normal_sample(mu, sig):
    return np.random.normal(mu, sig)

def multivariate_normal_sample(m, S):
    return multivariate_normal(m, S)

def normal_fst_derivative(y, f, s):
    return (y-f)/s

def normal_snd_derivative(y, f, s):
    return -1/s

"""
MultiNormal distribution
"""
def sample(name, args):
    if name == "multivariate_normal":
        return multivariate_normal_sample(*args)
    else:
        return normal_sample(*args)

def fst_derivative(name, args):
    return normal_fst_derivative(*args)

def snd_derivative(name, args):
    return normal_snd_derivative(*args)