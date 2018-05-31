import numpy as np

"""
SSVI_TF_d
Model for SSVI tensor factorization
"""
class SSVI_TF_d(object):
    def __init__(self, p_likelihood, q_posterior, p_prior, likelihood_type="normal"):
        self.p_likelihood = p_likelihood
        self.q_posterior  = q_posterior
        self.p_prior      = p_prior

"""
SSVI_TF_GME
Extension over SSVI_TF_d where we add another layer of Gaussian noise
"""
class SSVI_TF_GME(object):
    def __init__(self, p_likelihood, q_posterior, p_prior, likelihood_type="normal"):
        self.p_likelihood = p_likelihood
        self.q_posterior  = q_posterior
        self.p_prior      = p_prior


"""
Helper distribution class to keep track of a group of distributions
"""
class distribution(object):
    def __init__(self, type, dims, params, inits):
        """
        :param dims: How the group of distributions are distributed
                TODO:

        :param params: ("mean", "cov", ...)
                where dimi is the shape of the parameters
        :param inits: (p1, p2, ...)

        """
        self.type = type
        self.names = params
        self.dims = dims

        if dims == 1:
            self.params = inits
        else:
            ndim = len(dims) # Number of hidden matrices
            self.params  = [[] for _ in range(ndim)]

            for k in range(ndim): # Go through each dimension
                nrows = dims[k]   # Get the number of rows for the hidden matrixa
                for _ in range(nrows): # Go through each row
                    distr_params = [np.copy(inits[i]) for i in range(len(params))]
                    # Form the list of parameters for the ith row
                    # Append it to the kth dimension
                    self.params[k].append(tuple(distr_params))

    def update(self, d, i, change):
        self.params[d][i] = change

    def find(self, d, i):
        if self.dims == 1:
            return self.params
        return self.params[d][i]



