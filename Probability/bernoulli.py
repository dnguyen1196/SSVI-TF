import numpy as np
from Probability.ProbFun import sigmoid
from scipy.stats import bernoulli


class BernoulliDistribution(object):
    def __init__(self):
        return

    def __str__(self):
        return "BernoulliDistribution"

    def sample(self, m, k):
        return np.random.binomial(1, m, size=k)

    def pdf(self, y, m):
        return bernoulli.pmf((y+1)/2, sigmoid(m))

    def fst_derivative_pdf(self, y, m):
        return y * sigmoid(m) * (1 - sigmoid(m))

    def snd_derivative_pdf(self, y, m):
        s = sigmoid(m)
        return y * np.square(1 - s) * s - y * np.square(s) * (1 - s)

    def log_pdf(self, y, m):
        return y * (1 - sigmoid(y * m))

    def fst_derivative_log_pdf(self, y, m):
        return y * (1 - sigmoid(y * m))

    def snd_derivative_log_pdf(self, y, m):
        return -np.square(y) * sigmoid(y * m) * sigmoid(-y * m)


