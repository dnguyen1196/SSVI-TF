import numpy as np
from Probability.ProbFun import poisson_link, sigmoid
from scipy.stats import poisson
from scipy.misc import factorial


class PoissonDistribution(object):
    def __init__(self):
        return

    def __str__(self):
        return "PoissonDistribution"

    def sample(self, m, s=None, k=1):
        return np.random.poisson(m, size=k)

    def pdf(self, y, m, s=None):
        return poisson.pmf(y, poisson_link(m))

    def fst_derivative_pdf(self, y, m, s=None):
        pmf = self.pdf(y, poisson_link(m))
        s   = sigmoid(m)
        A   = poisson_link(m)
        return 1/factorial(y) * pmf * s * (y/ A - 1)

    def snd_derivative_pdf(self, y, m, s=None):
        pmf = self.pdf(y, poisson_link(m))
        s   = sigmoid(m)
        A   = poisson_link(m)
        pmf_prime = self.fst_derivative_pdf(y, m)

        temp1 = (y/A - 1) * (pmf_prime * s + s*(1-s)*pmf)
        temp2 = pmf * s * (-y)/np.square(A) * s
        return 1/factorial(y) * (temp1 + temp2)

    def log_pdf(self, y, m, s=None):
        return poisson.logpmf(y, poisson_link(m))

    def fst_derivative_log_pdf(self, y, f, s=None):
        sigma = sigmoid(f)
        A = poisson_link(f)
        res = sigma * (np.divide(y, A) - 1)
        return res

    def snd_derivative_log_pdf(self, y, f, s=None):
        sigma = sigmoid(f)
        A = poisson_link(f)
        temp1 = sigma * (1 - sigma) * (np.divide(y, A) - 1)
        temp2 = y * np.square(sigma) / np.square(A)
        res = temp1 - temp2
        return np.minimum(0, res)

