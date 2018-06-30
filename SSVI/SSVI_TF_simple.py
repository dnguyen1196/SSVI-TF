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

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.

    def initialize_di_Di_si(self):
        Di = np.zeros((self.D, self.D))
        di = np.zeros((self.D,))
        si = 0.
        return di, Di, si

    def estimate_di_Di_si(self, dim, i, coord, y, m, S):
        # if self.likelihood_type == "normal":
        #     return self.compute_di_Di_si_complete_conditional(dim, i, coord, y, m, S)
        return self.estimate_di_Di_si_by_sampling(dim, i, coord, y, m, S)

    def estimate_di_Di_si_by_sampling(self, dim, i, coord, y, m, S):
        othercols    = coord[: dim]
        othercols.extend(coord[dim + 1 :])

        alldims       = list(range(self.order))
        otherdims     = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        di          = np.zeros((self.D, ))
        Di          = np.zeros((self.D, self.D))

        for k1 in range(self.k1):
            vjs     = self.sample_vjs(othercols, otherdims)
            w       = np.random.rayleigh(self.w_sigma)
            meanf   = np.dot(vjs, m)
            covS    = np.dot(vjs, np.inner(S, vjs)) + w

            fst_deriv, snd_deriv = \
                self.estimate_expected_log_derivative(y, meanf, covS)

            di += vjs * fst_deriv/self.k1                   # Update di
            Di += np.outer(vjs, vjs) * snd_deriv/(2*self.k1) # Update Di

        si = self.estimate_si(othercols, otherdims, m, S, y)

        return di, Di, si

    def estimate_expected_log_derivative(self, y, meanf, covS) :
        first_derivative = 0.0
        snd_derivative = 0.0
        s = self.likelihood_param

        for k2 in range(self.k2):
            f = np.random.normal(meanf, covS)
            snd_derivative += self.likelihood.snd_derivative_log_pdf(y, f, s)
            first_derivative += self.likelihood.fst_derivative_log_pdf(y, f, s)

        return first_derivative/self.k2, snd_derivative/self.k2

    def estimate_si(self, othercols, otherdims, m, S, y):
        si = 0.
        for k1 in range(self.k1):
            ui  = np.random.multivariate_normal(m, S)
            uis = self.sample_vjs(othercols, otherdims)
            w   = np.random.rayleigh(self.w_sigma)

            meanf = np.sum(np.multiply(ui, uis))
            covf  = w
            _, snd_deriv = self.estimate_expected_log_derivative(y, meanf, covf)

            si += w/(8 * np.square(self.w_sigma)) * snd_deriv
        return si / self.k1

    def compute_di_Di_si_complete_conditional(self, dim, i, coord, y, mui, Sui):
        othercols    = coord[: dim]
        othercols.extend(coord[dim + 1 :])

        alldims       = list(range(self.order))
        otherdims     = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        d_acc = np.ones((self.D,))
        D_acc = np.ones((self.D,self.D))
        s = self.likelihood_param

        for j, d in enumerate(otherdims):
            m, S = self.posterior.get_vector_distribution(d, othercols[j])
            d_acc = np.multiply(d_acc, m)
            D_acc = np.multiply(D_acc, S + np.outer(m, m))

        Di = -1./(2*s) * D_acc
        di = y/s * d_acc - 1./s * np.inner(D_acc, mui)
        return di, Di