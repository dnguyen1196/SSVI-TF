from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv


class SSVI_TF_robust(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None, sigma0=1,k1=20, k2=10, batch_size=100):

        super(SSVI_TF_robust, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, sigma0,k1, k2, batch_size)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

    def initialize_di_Di_si(self):
        Di = np.zeros((self.D, self.D))
        di = np.zeros((self.D,))
        si = 0.
        return di, Di, si

    def estimate_di_Di_si(self, dim, i, coord, y, m, S):
        """
        :param dim:
        :param i:
        :param coord:
        :param y:
        :return:

        Estimate the di and Di (see paper for details) based
        on the given coordinate point, its value y, current
        mean and covariance m, S
        """
        othercols    = coord[: dim]
        othercols.extend(coord[dim + 1 :])

        alldims       = list(range(self.order))
        otherdims     = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        di          = np.zeros((self.D, ))
        Di          = np.zeros((self.D, self.D))
        si          = 0.

        for k1 in range(self.k1):
            ui = self.sample_vjs(othercols, otherdims)
            w  = np.random.rayleigh(np.square(self.w_sigma))

            meanf     = np.dot(ui, m)
            covS     = np.dot(ui, np.inner(S, ui)) + w
            pdf, pdf_1st, pdf_2nd = self.approximate_expected_derivative_pdf(y, meanf, covS)

            delta_phi = ui * pdf_1st
            delta_squared_phi = np.outer(ui, ui) * pdf_2nd
            delta_log_phi = 1/pdf * delta_phi
            delta_squared_log_phi = 1/ pdf * (delta_squared_phi - 1/ pdf * np.outer(delta_phi, delta_phi))

            di += 1/pdf * delta_phi
            Di += 1/2 * delta_squared_log_phi
            si += w/(8 * np.square(self.w_sigma)) * pdf_2nd/ self.k1 * 1/pdf

        return di, Di, si

    def approximate_expected_derivative_pdf(self, y, meanf, varf):
        fs = np.random.normal(meanf, varf, (self.k2,))
        pdf = 0.
        pdf_prime = 0.
        pdf_double_prime = 0.
        s = self.likelihood_param
        for fij in fs:
            pdf += 1/self.k2 * self.likelihood.pdf(y, fij, s)
            pdf_prime += 1/self.k2 * self.likelihood.fst_derivative_pdf(y, fij, s)
            pdf_double_prime += 1/self.k2 * self.likelihood.snd_derivative_pdf(y, fij, s)
        return pdf, pdf_prime, pdf_double_prime

    def estimate_expected_derivative(self, y, meanf, covS) :
        first_derivative = 0.0
        snd_derivative = 0.0
        s = self.likelihood_param

        for k2 in range(self.k2):
            f = np.random.normal(meanf, covS)
            snd_derivative += self.likelihood.snd_derivative_log_pdf(y, f, s)
            first_derivative += self.likelihood.fst_derivative_log_pdf(y, f, s)

        return first_derivative/self.k2, snd_derivative/self.k2

    def update_mean_param(self, dim, i, m, S, di_acc, Di_acc):
        raise NotImplementedError

    def update_cov_param(self, dim, i, m, S, di_acc, Di_acc):
        raise NotImplementedError

    def update_sigma_param(self, si_acc, scale):
        raise NotImplementedError

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        raise NotImplementedError