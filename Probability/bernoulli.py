import numpy as np
from Probability.ProbFun import sigmoid
from scipy.stats import bernoulli


class BernoulliDistribution(object):
    def __init__(self):
        return

    @staticmethod
    def pdf(self, y, m):
        return bernoulli.pmf((y+1)/2, sigmoid(m))

    @staticmethod
    def fst_derivative_pdf(self, y, m):
        return y * sigmoid(m) * (1 - sigmoid(m))

    @staticmethod
    def snd_derivative_pdf(self, y, m):
        s = sigmoid(m)
        return y * np.square(1 - s) * s - y * np.square(s) * (1 - s)

    @staticmethod
    def log_pdf(self, y, m):
        return y * (1 - sigmoid(y * m))

    @staticmethod
    def fst_derivative_log_pdf(self, y, m):
        return y * (1 - sigmoid(y * m))

    @staticmethod
    def snd_derivative_log_pdf(self, y, m):
        return -np.square(y) * sigmoid(y * m) * sigmoid(-y * m)


