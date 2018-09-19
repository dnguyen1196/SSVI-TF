from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv

class SSVI_TF_d(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 diag=False, mean0=None, cov0=None, sigma0=1, \
                 unstable_cov=False, k1=64, k2=64, batch_size=128, \
                 eta=1, cov_eta=1, sigma_eta=1, randstart=True):

        super(SSVI_TF_d, self).__init__(tensor, rank, mean_update, cov_update, noise_update, diag, \
                 mean0, cov0, sigma0, unstable_cov, k1, k2, batch_size, eta, cov_eta, sigma_eta, randstart)


    def estimate_expected_derivative_batch(self, ys, mean_batch, var_batch, ws_batch=None):
        num_samples     = np.size(mean_batch, axis=0)
        assert(self.k1 == np.size(mean_batch, axis=1))
        fst_deriv_batch = np.zeros((num_samples,self.k1))
        snd_deriv_batch = np.zeros((num_samples,self.k1))
        s  = self.likelihood_param
        #fs_report_mean = []
        #fs_report_std = []

        for num in range(num_samples):
            fs = np.random.normal(mean_batch[num, :], var_batch[num, :], size=(self.k2, self.k1))
            #fs_report_mean.append(np.mean(fs))
            #fs_report_std.append(np.std(fs))
            #fst_derivative_log_pdf = np.zeros((self.k2,self.k1))
            #for k in range(self.k2):
            #    fst_derivative_log_pdf = self.likelihood.fst_derivative_log_pdf(ys[num], fs[k, :], s)
            #fst_derivative_log_pdf = self.likelihood.fst_derivative_log_pdf(ys[num], fs, s)
            #snd_derivative_log_pdf = np.zeros((self.k2,self.k1))
            #for k in range(self.k2):
            #    snd_derivative_log_pdf[k, :] = self.likelihood.snd_derivative_log_pdf(ys[num], fs[k, :], s)
            fst_deriv_batch[num] = np.average(self.likelihood.fst_derivative_log_pdf(ys[num], fs, s),axis=0)
            #fst_deriv_batch[num] = self.likelihood.fst_derivative_log_pdf(ys[num], mean_batch[num, :], s)
            #for k in range(self.k1):
            #    v = fst_deriv_batch[num,k]
            #    if np.abs(v) < 0.1:
            #        #print("small")
            #        fst_deriv_batch[num, k] = np.mean(fst_deriv_batch[num, :])
            #snd_derivative_log_pdf = self.likelihood.snd_derivative_log_pdf(ys[num], fs, s)
            snd_deriv_batch[num] = np.average(self.likelihood.snd_derivative_log_pdf(ys[num], fs, s), axis=0)
        #print("expected deriv.fs", np.mean(np.asarray(fs_report_mean)), "std:", np.mean(np.asarray(fs_report_std)))
        #print("max-min", np.max(np.mean(fst_deriv_batch, axis=1)), np.min(np.mean(fst_deriv_batch, axis=1))) 
        return fst_deriv_batch, snd_deriv_batch, None


    def approximate_di_Di_si_without_second_layer_samplings(self, vjs_batch, ys, mean_batch):
        num_samples = np.size(mean_batch, axis=0)
        fst_deriv_batch = np.zeros((num_samples,self.k1))
        snd_deriv_batch = np.zeros((num_samples,self.k1))
        s  = self.likelihood_param
        for num in range(num_samples):
            fs = mean_batch[num, :]
            fst_deriv_batch[num] = np.average(self.likelihood.fst_derivative_log_pdf(ys[num], fs, s), axis=0)
            snd_deriv_batch[num] = np.average(self.likelihood.snd_derivative_log_pdf(ys[num], fs, s), axis=0)
        di = np.zeros((self.D,))

        #print("max-min", np.max(np.mean(fst_deriv_batch, axis=1)), np.min(np.mean(fst_deriv_batch, axis=1))) 
        if self.diag:
            Di = np.zeros((self.D,))
        else:
            Di = np.zeros((self.D,self.D))

        for num in range(num_samples):
            # Compute vj * scale
            vjs_batch_scaled = np.transpose(np.multiply(np.transpose(vjs_batch[num, :, :]), fst_deriv_batch[num, :]))
            di += np.average(vjs_batch_scaled, axis=0)
            for k1 in range(self.k1):
                vj = vjs_batch[num, k1, :]
                if self.diag:
                    Di += 0.5 * np.multiply(vj, vj) * snd_deriv_batch[num, k1]/self.k1
                else:
                    Di += 0.5 * np.outer(vj, vj) * snd_deriv_batch[num, k1] / self.k1
        return di, Di, None

    # TODO: implement closed form version
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        return self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)
