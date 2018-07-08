from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv

class SSVI_TF_d(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None, sigma0=1, k1=64, k2=64, batch_size=100):

        super(SSVI_TF_d, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, sigma0, k1, k2, batch_size)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

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
            snd_deriv_batch[num] = np.average(self.likelihood.snd_derivative_log_pdf(ys[num], fs, s), axis=0)

        return fst_deriv_batch, snd_deriv_batch

    # TODO: implement closed form version
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        return self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)