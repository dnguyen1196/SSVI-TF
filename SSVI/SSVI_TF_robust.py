from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv


class SSVI_TF_robust(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None,\
                 sigma0=1,k1=50, k2=50, batch_size=128, \
                 window_size=5, eta=1, cov_eta=1, sigma_eta=1, scheme="adagrad"):

        super(SSVI_TF_robust, self).__init__(tensor, rank, mean_update, cov_update, noise_update, \
                 mean0, cov0, \
                 sigma0,k1, k2, batch_size, eta, cov_eta, sigma_eta)

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)
        self.posterior       = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0)

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.

        self.noise_added = True

    def estimate_di_Di_si_batch(self, dim, i, coords, ys, m, S):
        num_subsamples     = np.size(coords, axis=0) # Number of subsamples

        othercols_left     = coords[:, : dim]
        othercols_right    = coords[:, dim + 1 :]
        othercols_concat   = np.concatenate((othercols_left, othercols_right), axis=1)

        alldims            = list(range(self.order))
        otherdims          = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        # Shape of vjs_batch would be (num_subsamples, k1, D)
        # Note that the formulation for fully robust model requires sampling for all
        # component vectors

        vjs_batch = self.sample_vjs_batch(othercols_concat, otherdims, self.k1)
        # print("vjs ", vjs_batch.shape)

        # uis_batch.shape = (num_samples, k1, D)
        # ....
        # TODO: check dimensionality!!
        uis_batch = np.random.multivariate_normal(m, S, size=(num_subsamples,self.k1))

        assert(uis_batch.shape == vjs_batch.shape)

        # assert(num_subsamples == np.size(uis_batch, axis=0)) # sanity check
        # assert(num_subsamples == np.size(vjs_batch, axis=0)) # sanity check

        ws_batch   = np.random.rayleigh(np.square(self.w_sigma), size=(num_subsamples, self.k1))

        # mean_batch.shape = (num_samples, k1)
        mean_batch = np.sum(np.multiply(vjs_batch, uis_batch), axis=2)
        assert(mean_batch.shape == (num_subsamples, self.k1))

        di, Di, si = self.approximate_di_Di_si_with_second_layer_samplings(ys, mean_batch, vjs_batch, ws_batch)

        return di, Di, si

    def approximate_di_Di_si_with_second_layer_samplings(self, ys, mean_batch, vjs_batch, ws_batch):
        """

        :param ys:          (num_samples,)
        :param mean_batch:  (num_samples, k1)
        :param vjs_batch:   (num_samples, k1, D)
        :param ws_batch:    (num_samples, k1)
        :return:
        """
        num_samples     = np.size(mean_batch, axis=0)
        assert(self.k1 == np.size(mean_batch, axis=1))

        # All shapes are (num_samples, k1)
        phi, phi_fst, phi_snd = self.estimate_expected_derivatives_pdf_batch(ys, mean_batch, ws_batch)

        di = np.zeros((num_samples, self.D))
        Di = np.zeros((num_samples, self.D, self.D))
        si = np.zeros((num_samples,))

        for num in range(num_samples):
            # p.shape = (k1,)
            p = phi[num, :]
            p1 = phi_fst[num, :]
            p2 = phi_snd[num, :]

            v  = vjs_batch[num, :, :] # v.shape = (k1, D)
            w  = ws_batch[num, :]     # w.shape = (k1,)

            di[num, :] = np.mean(np.transpose(np.multiply(np.transpose(v), \
                                                             1/p * p1)), axis=0)

            si[num]    = np.mean(w * p2 / p) / (8*np.square(self.w_sigma))

            for k in range(self.k1):
                Di[num, :, :] += 0.5 / self.k1 / p[k] * \
                                 (np.outer(v[k, :], v[k, :]) * p2[k] \
                                  - 1/p[k] * np.outer(v[k, :]*p2[k], v[k, :]*p2[k]))

        di = np.sum(di, axis=0)
        Di = np.sum(Di, axis=0)
        si = np.sum(si)

        return di, Di, si

    def estimate_expected_derivatives_pdf_batch(self, ys, mean_batch, ws_batch):
        """
        :param ys:          (num_samples)
        :param mean_batch:  (num_samples, k1)
        :param ws_batch:    (num_samples, k1)
        :return:
        """
        # print("mean_batch: ", mean_batch.shape)
        # print("ys: ", ys.shape)

        s = self.likelihood_param

        num_samples = np.size(ys, axis=0)
        pdf         = np.zeros((num_samples, self.k1))
        fst_deriv   = np.zeros_like(pdf)
        snd_deriv   = np.zeros_like(pdf)

        # print("ws_batch.norm ", np.linalg.norm(ws_batch))

        for num in range(num_samples):
            # For each num_samples, fs.shape = (k2, k1)
            fs = np.random.normal(mean_batch[num], ws_batch[num], size=(self.k2, self.k1))

            pdf[num, :]       = np.mean(self.likelihood.pdf(ys[num], fs, s), axis=0)
            fst_deriv[num, :] = np.mean(self.likelihood.fst_derivative_pdf(ys[num], fs, s), axis=0)
            snd_deriv[num, :] = np.mean(self.likelihood.snd_derivative_pdf(ys[num], fs, s), axis=0)

        # assert (pdf.shape == (num_samples, self.k1))
        # assert (fst_deriv.shape == (num_samples, self.k1))
        # assert (snd_deriv.shape == (num_samples, self.k1))
        return pdf, fst_deriv, snd_deriv

    # TODO: implement closed form version
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        return self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)

    def compute_stepsize_mean_param(self, dim, i, mGrad):
        return self.compute_stepsize_mean_param_ada_grad(dim, i, mGrad)

    def compute_stepsize_mean_param_ada_grad(self, dim, i, mGrad):
        acc_grad = self.ada_acc_grad[dim][:, i]
        grad_sqr = np.square(mGrad)
        self.ada_acc_grad[dim][:, i] += grad_sqr

        # return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        if self.likelihood_type != "poisson":
            return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        else:
            return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def compute_stepsize_mean_param_ada_delta(self, dim, i, mGrad):
        if self.d_mean < 0.4:
            # When reaching sufficiently small d_mean, switch to ada grad
            del(self.ada_delta_grad)
            del(self.ada_delta_ptr)
            self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.dims]
            self.compute_stepsize_mean_param = self.compute_stepsize_mean_param_ada_grad
            self.eta = 0.1 # Also change eta?

            print("Switching to ada grad with eta = ", self.eta)
            return self.compute_stepsize_mean_param_ada_grad(dim, i, mGrad)

        delta_grads = self.ada_delta_grad[dim][i, :, :]

        cur = self.ada_delta_ptr[dim][i]
        ptr = cur % self.window_size
        delta_grads[:, ptr] = np.square(mGrad)

        if cur + 1 >= self.window_size:
            step_size = self.eta / np.sqrt(np.sum(delta_grads, axis=1))
        else:
            step_size = self.eta / np.sqrt(np.sum(delta_grads[:, : cur + 1], axis=1))
        # print("step size: ", step_size )
        self.ada_delta_ptr[dim][i] = cur + 1

        return step_size