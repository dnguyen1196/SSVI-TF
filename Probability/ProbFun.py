import numpy as np
from numpy.random import multivariate_normal
import warnings

warnings.filterwarnings("error")

"""
Probability functions, used for random sampling
as well as taking derivative of the various distributions
"""
def sigmoid(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        if x < 0:
            return 0.00000001
        else:
            return 0.99999999

def sigmoid_overflow(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        return 0.00001

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
Bernoulli distribution (binary-valued tensor)
"""
def bernoulli_sample(f, s=None):
    return np.random.binomial(1, sigmoid(f))

def bernoulli_fst_derivative(y, f, s=None):
    return y * (1 - sigmoid(y * f))

def bernoulli_snd_derivative(y, f, s=None):
    return -np.square(y) * sigmoid(y * f) * sigmoid(-y * f)


"""
Poisson distribution (count-valued tensor)
"""
def poisson_sample(f):
    return np.random.poisson(lam=f)

def poisson_fst_derivative(y, f):
    s = min(0, f)
    p = sigmoid(f)
    return (y+1)/2-p

def poisson_snd_derivative(y, f):
    s = min(0, f)
    return -np.exp(2*s-f)/np.square(np.exp(s)+np.exp(s-f))

"""
MultiNormal distribution
"""
def sample(name, args):
    if name == "multivariate_normal":
        return multivariate_normal_sample(*args)
    else:
        return normal_sample(*args)

def fst_derivative(name, args):
    if name == "normal":
        return normal_fst_derivative(*args)
    elif name == "poisson":
        return poisson_fst_derivative(*args)
    elif name == "bernoulli":
        return bernoulli_fst_derivative(*args)

def snd_derivative(name, args):
    if name == "normal":
        return normal_snd_derivative(*args)
    elif name == "poisson":
        return poisson_snd_derivative(*args)
    elif name == "bernoulli":
        return bernoulli_snd_derivative(*args)