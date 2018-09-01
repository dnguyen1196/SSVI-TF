import math
from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv, norm, eig, eigh

class SSVI_TF_robust(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 diag=False, mean0=None, cov0=None, sigma0=1, \
                 unstable_cov=False, k1=64, k2=64, batch_size=128, \
                 eta=1, cov_eta=1, sigma_eta=1, quadrature=False, randstart=True):

        super(SSVI_TF_robust, self).__init__(tensor, rank, mean_update, cov_update, noise_update, diag, \
                 mean0, cov0, sigma0, unstable_cov, k1, k2, batch_size, eta, cov_eta, sigma_eta, randstart)

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.
        self.noise_added = True

        self.numerical_epsilon = 1e-16

        self.quadrature = quadrature
        if quadrature:
            self.gauss_degree = 30
            self.xs, self.weights = np.polynomial.hermite.hermgauss(self.gauss_degree)

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

        # uis_batch.shape = (num_samples, k1, D)
        if self.diag:
            uis_batch = np.random.multivariate_normal(m, np.diag(S), size=(num_subsamples,self.k1))
        else:
            uis_batch = np.random.multivariate_normal(m, S, size=(num_subsamples,self.k1))

        # assert(uis_batch.shape == vjs_batch.shape)           # sanity check
        # assert(num_subsamples == np.size(uis_batch, axis=0)) # sanity check
        # assert(num_subsamples == np.size(vjs_batch, axis=0)) # sanity check
        ws_batch   = np.random.rayleigh(np.square(self.w_sigma), size=(num_subsamples, self.k1))

        # mean_batch.shape = (num_samples, k1)
        mean_batch = np.sum(np.multiply(vjs_batch, uis_batch), axis=2)
        # assert (mean_batch.shape == (num_subsamples, self.k1)) # sanity check

        di, Di, si = self.approximate_di_Di_si_with_second_layer_samplings(ys, mean_batch, vjs_batch, ws_batch)

        #assert(not np.any(np.iscomplex(di)))
        #assert(not np.any(np.iscomplex(Di)))
        #assert(not np.any(np.iscomplex(si)))

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
        if self.quadrature:
            phi, phi_fst, phi_snd = self.estimate_expected_derivatives_pdf_batch_quadrature(ys, mean_batch, ws_batch)
        else:
            phi, phi_fst, phi_snd = self.estimate_expected_derivatives_pdf_batch(ys, mean_batch, ws_batch)

        #assert(not np.any(np.iscomplex(phi)))
        #assert(not np.any(np.iscomplex(phi_fst)))
        #assert(not np.any(np.iscomplex(phi_snd)))
        assert(not np.any(np.isnan(phi)))
        assert(not np.any(np.isnan(phi_snd)))
        assert(not np.any(np.isnan(phi_fst)))
        assert(not np.any(np.isinf(phi)))
        assert(not np.any(np.isinf(phi_fst)))
        assert(not np.any(np.isinf(phi_snd)))

        di = np.zeros((num_samples, self.D))
        if self.diag:
            Di = np.zeros((num_samples, self.D))
        else:
            Di = np.zeros((num_samples, self.D, self.D))
        si = np.zeros((num_samples,))

        for num in range(num_samples):
            # p.shape == p1.shape == p2.shape == (k1,)
            p = phi[num, :]
            p1 = phi_fst[num, :]
            p2 = phi_snd[num, :]
            assert(not np.any(p == 0))
            v  = vjs_batch[num, :, :] # v.shape = (k1, D)
            w  = ws_batch[num, :]     # w.shape = (k1,)

            di[num, :] = np.mean(np.transpose(np.multiply(np.transpose(v), \
                                                             1/p * p1)), axis=0)

            si[num]    = np.mean(w * p2 / p) / (8*np.square(self.w_sigma))

            for k in range(self.k1):

                if self.diag:
                    pre = 0.5 / np.multiply(self.k1, p[k])
                    temp1 = np.multiply(np.multiply(v[k, :], v[k, :]), p2[k])

                    v_phi = np.multiply(v[k, :], p2[k])
                    temp2 = np.multiply(1/p[k], np.multiply(v_phi, v_phi))

                    Di[num, :] += np.multiply(pre, np.subtract(temp1, temp2))

                else:
                    pre = 0.5 / np.multiply(self.k1, p[k])
                    temp1 = np.multiply(np.outer(v[k, :], v[k, :]), p2[k])

                    v_phi = np.multiply(v[k, :], p2[k])
                    temp2 = np.multiply(1/p[k], np.outer(v_phi, v_phi))

                    Di[num, :, :] += np.multiply(pre, np.subtract(temp1, temp2))

        di = np.sum(di, axis=0)
        Di = np.sum(Di, axis=0)
        si = np.sum(si)

        #assert(not np.any(np.iscomplex(di)))
        #assert(not np.any(np.iscomplex(Di)))
        #assert(not np.any(np.iscomplex(si)))
        assert(not np.any(np.isnan(Di)))
        assert(not np.any(np.isinf(Di)))

        return di, Di, si

    def estimate_expected_derivatives_pdf_batch(self, ys, mean_batch, ws_batch):
        """
        :param ys:          (num_samples)
        :param mean_batch:  (num_samples, k1)
        :param ws_batch:    (num_samples, k1)
        :return:
        """
        s = self.likelihood_param

        num_samples = np.size(ys, axis=0)
        pdf         = np.zeros((num_samples, self.k1))
        fst_deriv   = np.zeros_like(pdf)
        snd_deriv   = np.zeros_like(pdf)

        for num in range(num_samples):
            # For each num_samples, fs.shape = (k2, k1)
            fs = np.random.normal(mean_batch[num], ws_batch[num], size=(self.k2, self.k1))

            pdf[num, :]       = np.mean(self.likelihood.pdf(ys[num], fs, s), axis=0)
            fst_deriv[num, :] = np.mean(self.likelihood.fst_derivative_pdf(ys[num], fs, s), axis=0)
            snd_deriv[num, :] = np.mean(self.likelihood.snd_derivative_pdf(ys[num], fs, s), axis=0)

        return pdf, fst_deriv, snd_deriv

    # TODO: implement closed form version if exists
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        return self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)

    def estimate_expected_derivatives_pdf_batch_quadrature(self, ys, mean_batch, ws_batch):
        """
        :param ys: (num_samples)
        :param mean_batch: (num_samples, k1)
        :param ws_batch: (num_samples, k1)

        return: approximation to the integral int(N(f|m,w) p(y|f))
        """
        num_samples = np.size(ys, axis=0)
        pdf         = np.zeros((num_samples, self.k1))
        fst_deriv   = np.zeros_like(pdf)
        snd_deriv   = np.zeros_like(pdf)

        s = self.likelihood_param
        for num in range(num_samples):
            ms = mean_batch[num, :] # shape = k1
            ws = ws_batch[num, :]   # shape = k1
            assert(ms.shape == (self.k1,))
            assert(ws.shape == (self.k1,))
            y  = ys[num]
            for k in range(self.k1):
                E_ps, E_fst, E_snd = self.estimate_expected_pdf_quadrature(y, ms[k], ws[k])
                pdf[num, k] = E_ps
                fst_deriv[num, k] = E_fst
                snd_deriv[num, k] = E_snd

        return pdf, fst_deriv, snd_deriv

    def estimate_expected_pdf_quadrature(self, y, m, w):
        s  = self.likelihood_param
        fs = m + np.sqrt(2* np.square(w)) * self.xs # fs.shape = (gauss_degree)
        # ps, ps_fst, ps_snd -> shape = (gauss_degree)
        ps     = self.likelihood.pdf(y, fs, s)
        ps_fst = self.likelihood.fst_derivative_pdf(y, fs, s)
        ps_snd = self.likelihood.snd_derivative_pdf(y, fs, s)

        E_ps = np.sum(self.weights * ps) / np.sqrt(np.pi)
        E_fst = np.sum(self.weights * ps_fst) / np.sqrt(np.pi)
        E_snd = np.sum(self.weights * ps_snd) / np.sqrt(np.pi)
        return E_ps, E_fst, E_snd

    def update_sigma_param(self, si_acc, scale):
        # print(si_acc)
        # print("scale: ", scale)
        si_acc *= scale
        w_grad = -1/(2 * np.square(self.w_tau)) + si_acc
        w_step = self.compute_stepsize_sigma_param(w_grad)
        # print("w_step: ", w_step)

        update = (1-w_step) * (-0.5/np.square(self.w_sigma)) + w_step * w_grad
        # print("update: ", update)

        next_sigma = np.sqrt(-0.5/update)

        if np.isnan(next_sigma):
            return self.w_sigma, w_grad

        return next_sigma, w_grad

    def estimate_expectation_term_vlb(self, entry, val):
        k = 20
        sampled_vectors_prod = np.ones((k, self.D))

        for dim, col in enumerate(entry):
            m, S = self.posterior.get_vector_distribution(dim, col)
            if self.diag:
                samples = np.random.multivariate_normal(m, np.diag(S), size=k)
            else:
                samples = np.random.multivariate_normal(m, S, size=k)
            sampled_vectors_prod *= samples

        ms = np.sum(sampled_vectors_prod, axis=1) # (k,)

        ws = np.random.rayleigh(np.square(self.w_sigma), (k,))
        fs = np.random.normal(ms, ws, size=(k,k))
        s = self.likelihood_param

        # pdf.shape = (k,k)
        pdf = self.likelihood.pdf(val, self.link_fun(fs), s)

        expected_pdf = np.mean(pdf, axis=1) # shape = (k,)

        #log_expected_pdf = math.log(expected_pdf)
        log_expected_pdf = np.log(expected_pdf.astype(float))
        e_term = np.mean(log_expected_pdf)
        return e_term






