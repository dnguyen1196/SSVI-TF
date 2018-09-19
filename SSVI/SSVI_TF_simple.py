from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv

class SSVI_TF_simple(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 diag=False, mean0=None, cov0=None, sigma0=1, \
                 unstable_cov=False, k1=64, k2=64, batch_size=128, \
                 eta=1, cov_eta=1, sigma_eta=1, randstart=True):

        super(SSVI_TF_simple, self).__init__(tensor, rank, mean_update, cov_update, noise_update, diag, \
                 mean0, cov0, sigma0, unstable_cov, k1, k2, batch_size, eta, cov_eta, sigma_eta, randstart)

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.
        self.noise_added = True

    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        return self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)

    def estimate_expected_derivative_batch(self, ys, mean_batch, var_batch, ws_batch):
        num_samples     = np.size(mean_batch, axis=0)

        assert(self.k1 == np.size(mean_batch, axis=1))

        fst_deriv_batch = np.zeros((num_samples,self.k1))
        snd_deriv_batch = np.zeros((num_samples,self.k1))
        si_batch        = np.zeros((num_samples,self.k1))

        s  = self.likelihood_param

        for num in range(num_samples):
            fs = np.random.normal(mean_batch[num, :], var_batch[num, :] + ws_batch[num, :], size=(self.k2, self.k1))
            fst_deriv_batch[num, :] = np.average(self.likelihood.fst_derivative_log_pdf(ys[num], fs, s), axis=0)
            snd_deriv_batch[num, :] = np.average(self.likelihood.snd_derivative_log_pdf(ys[num], fs, s), axis=0)
            si_batch[num, :] = np.multiply(snd_deriv_batch[num, :], ws_batch[num, :]) / (8 * np.square(self.w_sigma))
        si = si_batch.mean()
        return fst_deriv_batch, snd_deriv_batch, si

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
