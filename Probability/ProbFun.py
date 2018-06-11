import numpy as np
from numpy.random import multivariate_normal
import warnings

warnings.filterwarnings("error")

"""
Probability functions, used for random sampling
as well as taking derivative of the various distributions
"""
epsilon = 0.00000001

def sigmoid(x):
    try:
        res = 1./(1 + np.exp(-x))
        return res
    except Warning:
        if x < 0:
            return epsilon
        else:
            return 1 - epsilon

def poisson_link_no_exception(f):
    return np.log(1 + np.exp(f))

def poisson_link(f):
    try:
        res = np.log(1 + np.exp(f))
        return max(res, epsilon)
    except Warning:
        return f


"""
Normal distribution
"""
def normal_sample(mu, sig):
    return np.random.normal(mu, sig)

def multivariate_normal_sample(m, S):
    return multivariate_normal(m, S)

def normal_log_likelihood(y, f, s):
    print(y, f)
    return -0.5*np.log(2 * np.pi * np.sqrt(s)) - np.square(y-f)/(2*s)

def normal_fst_log_derivative(y, f, s):
    return (y-f)/s

def normal_snd_log_derivative(y, f, s):
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
    sigma = sigmoid(f)
    A     = poisson_link(f)
    res   = sigma * (np.divide(y, A) - 1)
    return res

def poisson_snd_derivative(y, f, s=None):
    sigma = sigmoid(f)
    A     = poisson_link(f)
    temp1 = sigma * (1-sigma) * (np.divide(y, A) - 1)
    temp2 = y * np.square(sigma)/np.square(A)
    res = temp1 - temp2
    return min(0, res)

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
        return normal_fst_log_derivative(*args)
    elif name == "poisson":
        return poisson_fst_derivative(*args)
    elif name == "bernoulli":
        return bernoulli_fst_derivative(*args)

def snd_derivative(name, args):
    if name == "normal":
        return normal_snd_log_derivative(*args)
    elif name == "poisson":
        return poisson_snd_derivative(*args)
    elif name == "bernoulli":
        return bernoulli_snd_derivative(*args)

def log_likelihood(name, args):
    if name == "normal":
        return normal_log_likelihood(*args)