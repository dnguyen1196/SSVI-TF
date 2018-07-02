from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv

class SSVI_TF_d(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None, sigma0=1,k1=30, k2=10, batch_size=100):

        super(SSVI_TF_d, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, sigma0,k1, k2, batch_size)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

    def initialize_di_Di_si(self):
        di = np.zeros((self.D,))
        Di = np.zeros((self.D, self.D))
        si = 0
        return (di, Di, si)

    def estimate_di_Di_si(self, dim, i, coord, y, m, S):
        if self.likelihood_type == "normal":
            di, Di = self.estimate_di_Di_complete_conditional(dim, i, coord, y, m, S)
        else:
            di, Di = self.estimate_di_Di(dim, i, coord, y, m, S)

        return di, Di, 0

    def update_sigma_param(self, si_acc, scale):
        return

    def estimate_di_Di_complete_conditional(self, dim, i, coord, y, mui, Sui):
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

    def estimate_di_Di(self, dim, i, coord, y, m, S):
        othercols    = coord[: dim]
        othercols.extend(coord[dim + 1 :])

        alldims       = list(range(self.order))
        otherdims     = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        di          = np.zeros((self.D, ))
        Di          = np.zeros((self.D, self.D))

        vjs_batch = self.sample_vjs_batch(othercols, otherdims, self.k1)

        for k1 in range(self.k1):
            ui = vjs_batch[k1, :]

            meanf    = np.dot(ui, m)
            covS     = np.dot(ui, np.inner(S, ui))

            fst_deriv, snd_deriv = \
                self.estimate_expected_derivative(y, meanf, covS)

            di += ui * fst_deriv/self.k1                   # Update di
            Di += np.outer(ui, ui) * snd_deriv/(2*self.k1) # Update Di

        return di, Di

    def estimate_expected_derivative(self, y, meanf, covS) :
        fst_deriv = 0.0
        snd_deriv = 0.0
        s = self.likelihood_param

        for k2 in range(self.k2):
            f = np.random.normal(meanf, covS)
            snd_deriv += self.likelihood.snd_derivative_log_pdf(y, f, s)
            fst_deriv += self.likelihood.fst_derivative_log_pdf(y, f, s)

        return fst_deriv/self.k2, snd_deriv/self.k2

    def approximate_di_Di_si_with_second_layer_samplings(self, vjs_batch, ys, mean_batch, cov_batch, ws_batch=None):
        num_samples = np.size(vjs_batch, axis=0)
        fst_deriv_batch, snd_deriv_batch = self.estimate_expected_derivative_batch(ys, mean_batch, cov_batch)

        di = np.zeros((self.D,))
        Di = np.zeros((self.D,self.D))

        for num in range(num_samples):
            # Compute vj * scale
            vjs_batch_scaled = np.transpose(np.multiply(np.transpose(vjs_batch[num, :, :]), fst_deriv_batch[num, :]))
            di += np.average(vjs_batch_scaled, axis=0)

            for k1 in range(self.k1):
                vj = vjs_batch[num, k1, :]
                Di += 1/2 * np.outer(vj, vj) * snd_deriv_batch[num, k1] / self.k1

        return di, Di, None

    def estimate_expected_derivative_batch(self, ys, mean_batch, cov_batch):
        num_samples     = np.size(mean_batch, axis=0)

        assert(self.k1 == np.size(mean_batch, axis=1))

        fst_deriv_batch = np.zeros((num_samples,self.k1))
        snd_deriv_batch = np.zeros((num_samples,self.k1))

        s  = self.likelihood_param

        for num in range(num_samples):
            fs = np.random.normal(mean_batch[num, :], cov_batch[num, :], size=(self.k2, self.k1))
            fst_deriv_batch[num] = np.average(self.likelihood.fst_derivative_log_pdf(ys[num], fs, s), axis=0)
            # print("fst " , fst_deriv_batch.shape)
            snd_deriv_batch[num] = np.average(self.likelihood.snd_derivative_log_pdf(ys[num], fs, s), axis=0)

        return fst_deriv_batch, snd_deriv_batch
