import numpy as np
from Probability.ProbFun import sigmoid
from scipy.stats import bernoulli


class BernoulliDistribution(object):
    def __init__(self):
        self.epsilon = 1e-8

    def __str__(self):
        return "BernoulliDistribution"

    def sample(self, m, s=None, k=1):
        return np.random.binomial(1, sigmoid(m), size=k)

    def pdf(self, y, m, s=None):
        res = sigmoid(np.multiply(y, m))
        # print(res)
        return res

    def fst_derivative_pdf(self, y, m, s=None):
        temp = np.multiply(y, sigmoid(np.multiply(y, m)))
        res  = np.multiply(temp, np.subtract(1, sigmoid(np.multiply(y, m))))
        return res

    def snd_derivative_pdf(self, y, m, s=None):
        s = sigmoid(y * m)
        # return y * np.square(1 - s) * s - y * np.square(s) * (1 - s)
        return s * (1-s) * (1 - 2*s)

    def log_pdf(self, y, m, s=None):
        return np.log(self.pdf(y, m, s))

    def fst_derivative_log_pdf(self, y, m, s=None):
        # ym = y * m
        # print("ym: ", ym.shape)
        # sym = sigmoid(-ym)
        # print("sym: ", sym.shape)
        # res =  y * (1. - sigmoid(y * m))
        # res =  y * sigmoid(y * m)
        res = np.multiply(y, sigmoid(-np.multiply(y, m)))
        return res

    def snd_derivative_log_pdf(self, y, m, s=None):
        # return -np.square(y) * sigmoid(y * m) * sigmoid(-y * m)
        return np.multiply(-sigmoid(np.multiply (y, m)), sigmoid(-np.multiply(y, m)))

