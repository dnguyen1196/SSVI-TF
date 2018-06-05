import numpy as np
from numpy.random import multivariate_normal
import warnings

warnings.filterwarnings("error")

"""
Probability functions, used for random sampling
as well as taking derivative of the various distributions
"""
offset = 0.00000001

def sigmoid(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        if x < 0:
            return offset
        else:
            return 1 - offset

def poisson_link(f):
    try:
        res = np.log(1 + np.exp(f))
        if res < offset:
            return offset
        else:
            return res

    except Warning:
        return f

def sigmoid_overflow(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        return offset

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
def poisson_sample(f, s=None):
    return np.random.poisson(f)

def poisson_fst_derivative(y, f, s=None):
    return sigmoid(f) * (y / poisson_link(f) - 1)

def poisson_snd_derivative(y, f, s=None):
    temp1 = sigmoid(f) * (1-sigmoid(f)) * (y / poisson_link(f) - 1)
    temp2 = np.square(sigmoid(f))/np.square(poisson_link(f))*y
    return temp1 + temp2

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