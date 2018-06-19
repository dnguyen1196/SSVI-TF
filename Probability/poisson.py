import numpy as np
from Probability.ProbFun import poisson_link, sigmoid
from scipy.stats import poisson
from scipy.misc import factorial

class PoissonDistribution(object):
    def __init__(self):
        return

    @staticmethod
    def pdf(self, y, m):
        return poisson.pmf(y, poisson_link(m))

    @staticmethod
    def fst_derivative_pdf(self, y, m):
        pmf = self.pdf(y, poisson_link(m))
        s   = sigmoid(m)
        A   = poisson_link(m)
        return 1/factorial(y) * pmf * s * (y/ A - 1)

    @staticmethod
    def snd_derivative_pdf(self, y, m):
        pmf = self.pdf(y, poisson_link(m))
        s   = sigmoid(m)
        A   = poisson_link(m)
        pmf_prime = self.fst_derivative_pdf(y, m)
        temp1 = (y/A - 1) * (pmf_prime * s + s*(1-s)*pmf)
        temp2 = pmf * s * (-y)/np.square(A) * s
        return 1/factorial(y) * (temp1 + temp2)

    @staticmethod
    def log_pdf(self, y, m):
        return poisson.logpmf(y, poisson_link(m))

    @staticmethod
    def fst_derivative_log_pdf(self, y, f):
        sigma = sigmoid(f)
        A = poisson_link(f)
        res = sigma * (np.divide(y, A) - 1)
        return res

    @staticmethod
    def snd_derivative_log_pdf(self, y, f):
        sigma = sigmoid(f)
        A = poisson_link(f)
        temp1 = sigma * (1 - sigma) * (np.divide(y, A) - 1)
        temp2 = y * np.square(sigma) / np.square(A)
        res = temp1 - temp2
        return min(0, res)

