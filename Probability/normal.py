import numpy as np
from scipy.stats import norm


class NormalDistribution(object):
    def __init__(self):
        return

    @staticmethod
    def pdf(self, y, m, s):
        return norm.pdf(y, m, s)

    @staticmethod
    def fst_derivative_pdf(self, y, m, s):
        return -norm.pdf(y, m, s) * (y - m) / s

    @staticmethod
    def snd_derivative_pdf(self, y, m, s):
        return norm.pdf(y, m, s) * (np.square((y-m)/s) - 1/s)

    @staticmethod
    def log_pdf(self, y, m, s):
        return np.log(norm.pdf(y, m, s))

    @staticmethod
    def fst_derivative_log_pdf(self, y, m, s):
        return 1/np.square(s) * (y - m)

    @staticmethod
    def snd_derivative_log_pdf(self, y, m, s):
        return -1/np.square(s)


