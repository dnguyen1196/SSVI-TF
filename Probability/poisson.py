import numpy as np
from Probability.ProbFun import poisson_link, sigmoid
from scipy.stats import poisson
from scipy.special import factorial

class PoissonDistribution(object):
    def __init__(self):
        self.epsilon = 1e-8

    def __str__(self):
        return "PoissonDistribution"

    def sample(self, m, s=None, k=1):
        return np.random.poisson(m, size=k)

    def pdf(self, y, m, S=None):
        A = poisson_link(m)
        # res = np.divide(np.multiply(np.power(A, y), np.exp(-A)), factorial(y))
        res = poisson.pmf(y, A)

        #res = np.maximum(res, self.epsilon)
        # print("res: ", res)
        return res

    def fst_derivative_pdf(self, y, m, S=None):
        # pdf * sigmoid * (y/A(f) - 1)
        A    = poisson_link(m)
        pdf  = self.pdf(y, m, S)
        sigm = sigmoid(m)

        assert(sigm.shape == m.shape)
        assert(A.shape == m.shape)

        temp = np.subtract(np.divide(y, A), 1)
        res  = np.multiply(pdf, np.multiply(sigm, temp))
        return res

    def snd_derivative_pdf(self, y, m, s=None):
        pdf = self.pdf(y, poisson_link(m))
        sigm   = sigmoid(m)
        A   = poisson_link(m)

        pmf_prime = self.fst_derivative_pdf(y, m)

        assert(sigm.shape == m.shape)
        assert(A.shape == m.shape)

        # temp1 = (y/A - 1) * (pmf_prime * sigm + sigm*(1-sigm)*pmf)
        # temp2 = pmf * sigm * (-y)/np.square(A) * sigm
        # return 1/factorial(y) * (temp1 + temp2)

        temp1 = np.multiply(pmf_prime, np.multiply(sigm, np.divide(y, A) - 1))

        temp  = np.multiply(-np.divide(y, np.square(A)), np.square(sigm)) \
                + np.multiply(y / A - 1, np.multiply(sigm, 1 - sigm))

        temp2 = np.multiply(pdf, temp)

        return temp1 + temp2

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

