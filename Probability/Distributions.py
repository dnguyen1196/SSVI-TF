import numpy as np

"""
Distributions.py
Contains the definition of some of the distributions

"""

"""
Poisson
"""
class Poisson(object):
    def __init__(self, l):
        self.l = l



"""
Normal
"""
class Normal(object):
    def __init__(self, m, S):
        self.m = m
        self.s = S

    def sample(self):
        return

    def fst_derivative(self, y, arg):
        return 1/self.s*(y-self.m)

    def snd_derivative(self, y, arg):
        return -1 / self.s


"""
MultiNormal
"""
class MultiNormal(object):
    """
    Normal(m, S, k)
    Args:
        - m := mean
        - S := covariance
        - k := dimension
    """
    def __init__(self, m, S, k, invS=True):
        self.m = m
        self.invS = invS
        self.k = k

        if invS:
            self.S = S
        else:
            self.S = np.linalg.inv(S)

        self.etaM = self.S * self.m
        self.etaS = -0.5 * self.S

    """
    sample()
    Generate a vector from the distribution
    """
    def sample(self):
        return

    """
    fst_derivative
    Find the first derivate with respect to which argument
    """
    def fst_derivative(self, arg):
        return

    """
    snd_derivative
    
    """
    def snd_derivate(self, arg):
        return



