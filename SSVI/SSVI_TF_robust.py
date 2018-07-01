from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv


class SSVI_TF_robust(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None, sigma0=1,k1=50, k2=50, batch_size=100, eta=0.05, cov_eta=.000001):

        super(SSVI_TF_robust, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, sigma0,k1, k2, batch_size)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.

        # Override the eta constant in adagrad
        self.eta = eta
        self.cov_eta = cov_eta
        self.noise_added = True

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

        vjs_batch = self.sample_vjs_batch(othercols, otherdims, self.k1)

        for k1 in range(self.k1):
            vjs = vjs_batch[k1, :]
            w   = np.random.rayleigh(np.square(self.w_sigma))

            meanf     = np.dot(vjs, m)
            covS      = w
            pdf, pdf_1st, pdf_2nd = self.approximate_expected_derivative_pdf(y, meanf, covS)

            inv_phi = np.divide(1, pdf)
            vj_phi_prime = np.multiply(vjs, pdf_1st)

            di += inv_phi * vj_phi_prime

            Di += inv_phi * (np.outer(vjs, vjs) * pdf_2nd \
                  - inv_phi * np.outer(vj_phi_prime, vj_phi_prime))

            si += np.divide(w, (8 * np.square(self.w_sigma))) * pdf_2nd * inv_phi

        return di/self.k1, 0.5 * Di/self.k1, si/self.k1

    def approximate_expected_derivative_pdf(self, y, meanf, varf):
        fs = np.random.normal(meanf, varf, (self.k2,))
        pdf = 0.
        pdf_prime = 0.
        pdf_double_prime = 0.
        s = self.likelihood_param
        for f in fs:
            pdf += self.likelihood.pdf(y, f, s)
            pdf_prime += self.likelihood.fst_derivative_pdf(y, f, s)
            pdf_double_prime += self.likelihood.snd_derivative_pdf(y, f, s)
        return pdf/self.k2, pdf_prime/self.k2, pdf_double_prime/self.k2

    # def compute_stepsize_cov_param(self, dim, i, covGrad):
    #     if self.likelihood_type == "poisson":
    #         return 1/(self.time_step[dim] + self.cov_stepsize_pois)
    #     return 0.001/(self.time_step[dim] + self.cov_stepsize)

