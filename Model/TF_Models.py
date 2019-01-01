import numpy as np
from abc import abstractclassmethod, abstractmethod


class Posterior(object):
    def __init__(self, dims, D, initMean=None, initCov=None):
        self.dims = dims
        self.D    = D
        self.initMean = initMean
        self.initCov  = initCov

        self.params = [[] for _ in self.dims]

        if initMean is None or initCov is None:
            self.initMean = np.ones((self.D,)) 
            self.initCov  = np.eye(self.D)

        for i, s in enumerate(self.dims):
            self.params[i] = self.initialize_params(s, self.initMean, self.initCov)

    @abstractmethod
    def initialize_params(self, nparams, mean, cov):
        raise NotImplementedError

    @abstractmethod
    def get_vector_distribution(self, dim, i):
        raise NotImplementedError

    @abstractmethod
    def update_vector_distribution(self, dim, i, m_next, S_next):
        raise NotImplementedError

    def save_mean_params(self, dim, filename):
        np.savetxt(filename, self.params[dim][:, 0, :], delimiter=",")

    def reset(self):
        self.params = [[] for _ in self.dims]
        for i, s in enumerate(self.dims):
            self.params[i] = self.initialize_params(s, self.initMean, self.initCov)

    def save_mean_parameters(self, dim,  mean_file):
        np.save(mean_file, self.params[dim][:, 0, :], delimiter=",")

    def save_cov_parameters(self, dim, cov_file):
        np.save(cov_file, self.params[dim][:, 1:, :], delimiter=",")


class Posterior_Full_Covariance(Posterior):
    def __init__(self, dims, D, initMean=None, initCov=None, randstart=True):
        self.randstart = randstart
        super(Posterior_Full_Covariance, self).__init__(dims, D, initMean, initCov)

    def initialize_params(self, nparams, mean, cov):
        matrices = np.zeros((nparams, self.D + 1, self.D))
        for i in range(nparams):
            if not self.randstart:
                matrices[i, 0, :]   = np.copy(mean)
                matrices[i, 1 :, :] = np.copy(cov)
            else:
                matrices[i, 0, :]   = np.random.multivariate_normal(mean, cov)
                matrices[i, 1 :, :]   = np.copy(cov)
        return matrices

    def get_vector_distribution(self, dim, i):
        return self.params[dim][i, 0, :], self.params[dim][i, 1 : , :]

    def update_vector_distribution(self, dim, i, m_next, S_next):
        self.params[dim][i, 0, :] = m_next
        self.params[dim][i, 1 : , :] = S_next


class Posterior_Diag_Covariance(Posterior):
    def __init__(self, dims, D, initMean=None, initCov=None, randstart=True):
        self.randstart = randstart
        super(Posterior_Diag_Covariance, self).__init__(dims, D, initMean, initCov)

    def initialize_params(self, nparams, mean, cov):
        matrices = np.zeros((nparams, 2, self.D))
        for i in range(nparams):
            if not self.randstart:
                matrices[i, 0, :]  = np.copy(mean)
                matrices[i, 1, :]  = np.copy(np.diag(cov))
            else:
                matrices[i, 0, :] = np.random.multivariate_normal(mean, cov)
                matrices[i, 1, :] = np.copy(np.diag(cov))
        return matrices

    def get_vector_distribution(self, dim, i):
        return self.params[dim][i, 0, :], self.params[dim][i, 1, :]

    def update_vector_distribution(self, dim, i, m_next, S_next):
        self.params[dim][i, 0, :] = m_next
        self.params[dim][i, 1, :] = S_next
