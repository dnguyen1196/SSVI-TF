from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv


class SSVI_TF_simple(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None, sigma0=1,k1=20, k2=10, batch_size=100):
        super(SSVI_TF_simple, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, sigma0,k1, k2, batch_size)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

    def initialize_di_Di_si(self):
        raise NotImplementedError

    def estimate_di_Di_si(self, dim, i, coord, y, m, S):
        raise NotImplementedError

    def update_mean_param(self, dim, i, m, S, di_acc, Di_acc):
        raise NotImplementedError

    def update_cov_param(self, dim, i, m, S, di_acc, Di_acc):
        raise NotImplementedError

    def update_sigma_param(self, si_acc, scale):
        raise NotImplementedError

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        raise NotImplementedError