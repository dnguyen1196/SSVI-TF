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
        return self.estimate_di_Di_si_sampling(dim, i, coord, y, m, S)

    def estimate_di_Di_si_sampling(self, dim, i, coord, y, m, S):
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

    def update_natural_params(self, dim, i, iteration):
        """
        :param i:
        :param dim:
        :return:
        Update the natural parameter for the hidden column vector
        of dimension dim and column i
        """
        observed_i = self.tensor.find_observed_ui(dim, i)
        # print(observed_i)
        if len(observed_i) > self.batch_size:
            observed_idx = np.random.choice(len(observed_i), self.batch_size, replace=False)
            observed_i = np.take(observed_i, observed_idx, axis=0)

        M = len(observed_i)
        (m, S) = self.posterior.get_vector_distribution(dim, i)

        Di_acc = np.zeros((self.D, self.D))
        di_acc = np.zeros((self.D,))
        si_acc = 0.

        for entry in observed_i:
            coord = entry[0]
            y = entry[1]

            # if self.likelihood_type == "normal": # Case of complete conditional
            #     (dij, Dij) = self.compute_di_Di_si_complete_conditional(dim, i, coord, y, m, S)
            # else:
            (dij, Dij, sij) = self.estimate_di_Di_si_sampling(dim, i, coord, y, m, S)

            Di_acc += Dij
            di_acc += dij
            si_acc += sij

        scale = len(observed_i) / min(self.batch_size, len(observed_i))

        Di_acc *= scale
        di_acc *= scale
        si_acc *= scale

        # Compute next covariance and mean
        S_next   = self.update_cov_param(dim, i, m, S, di_acc, Di_acc)
        m_next   = self.update_mean_param(dim, i, m, S, di_acc, Di_acc)

        self.update_sigma_param(si_acc, scale)

        # Measures the change in the parameters from previous iterations
        self.keep_track_changes_params(dim, i, m, S, m_next, S_next)

        # Update the change
        self.posterior.update_vector_distribution(dim, i, m_next, S_next)

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

    def update_cov_param(self, dim, i, m, S, di_acc,  Di_acc):
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

    def compute_stepsize_cov_param(self, dim, i, covGrad):
        if self.likelihood_type == "poisson":
            return 1/(self.time_step[dim]+100)
        return 1/(self.time_step[dim] + 1)

    def compute_stepsize_mean_param(self, dim, i, mGrad):
        """
        :param dim: dimension of the hidden matrix
        :param i: column of hidden matrix
        :param mGrad: computed gradient
        :return:

        Compute the update for the mean parameter dependnig on the
        optimization scheme
        """
        acc_grad = self.ada_acc_grad[dim][:, i]
        grad_sqr = np.square(mGrad)
        self.ada_acc_grad[dim][:, i] += grad_sqr

        # return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        if self.likelihood_type != "poisson":
            return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        else:
            return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def update_sigma_param(self, si_acc, scale):
        w_grad = -1/np.square(self.w_tau) + si_acc
        self.w_ada_grad += np.square(w_grad)
        w_step = self.eta / self.w_ada_grad
        update = (1-w_step) * (-1/np.square(self.w_sigma)) + w_step * w_grad
        self.w_sigma = np.sqrt(-1/update)

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        cov_change  = np.linalg.norm(S_next - S, 'fro')
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        cov_change  = np.linalg.norm(S_next - S, 'fro')
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])

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