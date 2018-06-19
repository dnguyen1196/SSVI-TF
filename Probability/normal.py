import numpy as np
from scipy.stats import norm

class NormalDistribution(object):
    def __init__(self):
        return

    def pdf(self, y, m, s):
        return norm.pdf(y, m, s)

    def fst_derivative_pdf(self, y, m, s):
        return -norm.pdf(y, m, s) * (y - m) / s

    def snd_derivative_pdf(self, y, m, s):
        return norm.pdf(y, m, s) * (np.square((y-m)/s) - 1/s)

    def log_pdf(self, y, m, s):
        return np.log(norm.pdf(y, m, s))

    def fst_derivative_log_pdf(self, y, m, s):
        return 1/np.square(s) * (y - m)

    def snd_derivative_log_pdf(self, y, m, s):
        return -1/np.square(s)

