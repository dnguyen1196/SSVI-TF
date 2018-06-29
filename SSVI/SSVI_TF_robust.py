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

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.

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
            vjs = self.sample_vjs(othercols, otherdims)
            w  = np.random.rayleigh(np.square(self.w_sigma))

            meanf     = np.dot(vjs, m)
            covS     = np.dot(vjs, np.inner(S, vjs)) + w
            pdf, pdf_1st, pdf_2nd = self.approximate_expected_derivative_pdf(y, meanf, covS)

            # print(vjs)
            # print(pdf_1st)
            # print(pdf)
            # print(pdf_2nd)

            delta_phi = vjs * pdf_1st
            delta_squared_phi = np.outer(vjs, vjs) * pdf_2nd
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
        for f in fs:
            print("y = ", y, "fij = ", f, " meanf = ", meanf, " varf = ", varf)
            print("pdf = ", self.likelihood.pdf(y, f, s), "pdf' = ", self.likelihood.fst_derivative_pdf(y, f, s))
            pdf += 1/self.k2 * self.likelihood.pdf(y, f, s)
            pdf_prime += 1/self.k2 * self.likelihood.fst_derivative_pdf(y, f, s)
            pdf_double_prime += 1/self.k2 * self.likelihood.snd_derivative_pdf(y, f, s)
        return pdf, pdf_prime, pdf_double_prime

    def update_mean_param(self, dim, i, m, S, di_acc, Di_acc):
        if self.mean_update == "S":
            meanGrad = (np.inner(self.pSigma_inv, self.pmu - m) + di_acc)
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            m_next = m + np.multiply(meanStep, meanGrad)
        elif self.mean_update == "N":
            C = np.inner(inv(S), m)
            meanGrad = np.inner(self.pSigma_inv, self.pmu) + di_acc - 2 * np.inner(Di_acc, m)
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            C_next   = np.multiply(1 -  meanStep, C) + meanStep * meanGrad
            m_next   = np.inner(S, C_next)
        else:
            raise Exception("Unidentified update formula for covariance param")
        return m_next

    def update_cov_param(self, dim, i, m, S, di_acc, Di_acc):
        if self.cov_update == "S":
            L = cholesky(S)
            covGrad = np.triu(inv(np.multiply(L, np.eye(self.D))) \
                      - np.inner(L, self.pSigma_inv) + 2 * np.inner(L, Di_acc))
            covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
            L_next  = L + covStep * covGrad
            S_next  = np.inner(L, np.transpose(L))
        elif self.cov_update == "N":
            covGrad = (self.pSigma_inv - 2 * Di_acc)
            covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
            S_next = inv((1 - covStep) * inv(S) + np.multiply(covStep, covGrad))
        else:
            raise Exception("Unidentified update formula for covariance param")
        return S_next

    def update_sigma_param(self, si_acc, scale):
        w_grad = -1/(2 * np.square(self.w_tau)) + 1/ (4 * np.square(self.w_sigma)) * si_acc
        w_step = self.compute_stepsize_sigma_param(w_grad)
        update = (1-w_step) * (-1/np.square(self.w_sigma)) + w_step * w_grad
        self.w_sigma = np.sqrt(-1/update)

    def compute_stepsize_mean_param(self, dim, i, mGrad):
        acc_grad = self.ada_acc_grad[dim][:, i]
        grad_sqr = np.square(mGrad)
        self.ada_acc_grad[dim][:, i] += grad_sqr

        # return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        if self.likelihood_type != "poisson":
            return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        else:
            return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def compute_stepsize_cov_param(self, dim, i, covGrad):
        if self.likelihood_type == "poisson":
            return 1/(self.time_step[dim]+100)
        return 1/(self.time_step[dim] + 1)

    def compute_stepsize_sigma_param(self, w_grad):
        self.w_ada_grad += np.square(w_grad)
        w_step = self.eta / self.w_ada_grad
        return w_step

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        cov_change  = np.linalg.norm(S_next - S, 'fro')
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])