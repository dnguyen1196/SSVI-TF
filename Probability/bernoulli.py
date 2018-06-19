import numpy as np
from scipy.stats import norm

class BernoulliDistribution(object):
    def __init__(self):
        return

    def pdf(self, y, m, s):
        return norm.pdf(y, m, s)

    def fst_derivative_pdf(self, y, m, s):
        return

    def snd_derivative_pdf(self, y, m, s):
        return

    def log_pdf(self, y, m, s):
        return

    def fst_derivative_log_pdf(self, y, m, s):
        return

    def snd_derivative_log_pdf(self, y, m, s):
        return


