from SSVI.SSVI_TF_interface import SSVI_TF
from Model.TF_Models import Posterior_Full_Covariance
import numpy as np
from numpy.linalg import inv, norm, eig, eigh


class SSVI_TF_robust(SSVI_TF):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                 mean0=None, cov0=None,\
                 sigma0=1,k1=50, k2=50, batch_size=128, \
                 eta=1, cov_eta=1, sigma_eta=1):

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

        # print("phi_prime/phi: ", phi_fst/ phi)
        # print("phi_snd/phi: ", phi_snd/ phi)

        di = np.zeros((num_samples, self.D))
        Di = np.zeros((num_samples, self.D, self.D))
        si = np.zeros((num_samples,))

        for num in range(num_samples):
            # print("sigma = ", self.w_sigma)
            # p.shape = (k1,)
            p = phi[num, :]
            p1 = phi_fst[num, :]
            p2 = phi_snd[num, :]

            v  = vjs_batch[num, :, :] # v.shape = (k1, D)
            w  = ws_batch[num, :]     # w.shape = (k1,)

            # print("nan: ", np.any(np.isnan(p2/p)))
            # print("p2/ p ", np.any(p == 0))
            # print("zro: ", np.any(p2/p == 0))
            #
            # if norm(np.square(p2)/p1) > 1:
            #     print("num: ", num)
            #     print("ys[num] ", ys[num])
            #     print("mean_batch ", mean_batch)

            di[num, :] = np.mean(np.transpose(np.multiply(np.transpose(v), \
                                                             1/p * p1)), axis=0)

            si[num]    = np.mean(w * p2 / p) / (8*np.square(self.w_sigma))
            # print("si[num] ", si[num])

            for k in range(self.k1):
                pre = 0.5 / np.multiply(self.k1, p[k])
                temp1 = np.multiply(np.outer(v[k, :], v[k, :]), p2[k])

                v_phi = np.multiply(v[k, :], p2[k])
                temp2 = np.multiply(1/p[k], np.outer(v_phi, v_phi))

                Di[num, :, :] += np.multiply(pre, np.subtract(temp1, temp2))

        di = np.sum(di, axis=0)
        Di = np.sum(Di, axis=0)
        si = np.sum(si)

        # print("di: ", np.linalg.norm(di))
        # print("Di: ", np.linalg.norm(Di, "fro"))
        # print("si: ", si)

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

    # def update_cov_param(self, dim, i, m, S, di_acc,  Di_acc):
    #     if self.cov_update == "S":
    #         L = cholesky(S)
    #         covGrad = self.remove_negative_eigenvals(\
    #                   np.triu(inv(np.multiply(L, np.eye(self.D))) \
    #                   - np.inner(L, self.pSigma_inv) + 2 * np.dot(L, Di_acc)))
    #
    #         covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
    #         L_next  = L + covStep * covGrad
    #         S_next  = np.dot(L, np.transpose(L))
    #
    #     elif self.cov_update == "N":
    #         covGrad = self.remove_negative_eigenvals(self.pSigma_inv - 2 * Di_acc)
    #         covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
    #         S_next = inv((1 - covStep) * inv(S) + np.multiply(covStep, covGrad))
    #
    #     else:
    #         raise Exception("Unidentified update formula for covariance param")
    #     return S_next
    #
    # def remove_negative_eigenvals(self, covGrad):
    #     return covGrad
    #
    #     eigvals, eigvec = eig(covGrad)
    #     eigvals    = np.maximum(np.real(eigvals), 0.)
    #     # w_pos  = np.minimum(np.real(w), 0.)
    #
    #     eigvals    = self.scale_eigenvals(eigvals)
    #     # print("ev: ", eigvals)
    #     eigvec     = np.real(eigvec)
    #     w_pos_vt   = np.multiply(np.transpose(eigvec), eigvals[:, np.newaxis])
    #
    #     # w_pos_vt = np.dot(np.diag(w), np.transpose(v))
    #     # print(norm(w_pos_vt))
    #     return np.dot(eigvec, w_pos_vt)
    #
    # def scale_eigenvals(self, eigvals):
    #     # return eigvals/norm(eigvals)
    #     return eigvals

    def predict_entry(self, entry):
        if self.likelihood_type == "normal":
            u = np.ones((self.D,))
            for dim, col in enumerate(entry):
                m, _ = self.posterior.get_vector_distribution(dim, col)
                u = np.multiply(u, m)
            return np.sum(u)

        res = self.estimate_expected_observation_sampling(entry)
        if self.likelihood_type == "bernoulli":
            return 1 if res > 1/2 else -1

        elif self.likelihood_type == "poisson":
            return res

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
            return self.w_sigma

        return next_sigma