from abc import abstractclassmethod, abstractmethod
import Probability.ProbFun as probs
import numpy as np

from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import cholesky
from Model.TF_Models import Posterior_Full_Covariance

from Probability.normal import NormalDistribution
from Probability.bernoulli import BernoulliDistribution
from Probability.poisson import PoissonDistribution

import math
import time

class SSVI_TF(object):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                        mean0=None, cov0=None, sigma0=1,k1=30, k2=10, batch_size=128, \
                        eta=1, cov_eta=1, sigma_eta=1):

        self.tensor = tensor
        self.dims   = tensor.dims
        self.datatype   = tensor.datatype
        self.order      = len(tensor.dims)   # number of dimensions

        self.D      = rank
        self.mean_update = mean_update
        self.cov_update  = cov_update
        self.noise_update = noise_update

        self.mean0  = mean0
        self.cov0   = cov0
        self.w_sigma = sigma0
        self.k1     = k1
        self.k2     = k2
        self.batch_size = batch_size

        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))

        # self.pSigma_inv      = np.eye(self.D)
        if self.datatype == "real":
            self.link_fun = lambda m : m
            self.likelihood_type = "normal"
            self.likelihood_param = 1
            self.likelihood = NormalDistribution()
        elif self.datatype == "binary":
            self.link_fun = lambda m : probs.sigmoid(m)
            self.likelihood_type = "bernoulli"
            self.likelihood_param = 0 # Not used
            self.likelihood = BernoulliDistribution()
        elif self.datatype == "count":
            self.likelihood_type = "poisson"
            self.max_count   = tensor.max_count
            self.min_count   = tensor.min_count
            self.herm_degree = 50
            self.hermite_points, self.hermite_weights = np.polynomial.hermite.hermgauss(self.herm_degree)
            self.link_fun = lambda m : probs.poisson_link(m)
            self.likelihood_param = 0 # Not used
            self.likelihood = PoissonDistribution()

        self.time_step = [0 for _ in range(self.order)]
        self.epsilon = 0.0001

        # Optimization parameter
        # self.scheme = scheme
        # if scheme == "adagrad":
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.dims]

        self.eta = eta
        self.cov_eta = cov_eta
        self.poisson_eta = 0.1
        self.sigma_eta = sigma_eta

        # keep track of changes in norm
        self.norm_changes = [np.zeros((s, 2)) for s in self.dims]
        self.noise_added  = False

    def factorize(self, report=1000):
        self.report = report
        update_column_pointer = [0] * self.order
        start = time.time()
        iteration = 0

        while True:
            current = time.time()

            for dim in range(self.order):
                col = update_column_pointer[dim]
                # Update the natural params of the col-th factor
                # in the dim-th dimension

                # self.update_natural_params(dim, col, iteration)
                self.update_natural_param_batch(dim, col, iteration)

                self.update_hyper_parameter(dim, iteration)

            # Move on to the next column of the hidden matrices
            for dim in range(self.order):
                if (update_column_pointer[dim] + 1 == self.dims[dim]):
                    self.time_step[dim] += 1  # increase time step
                update_column_pointer[dim] = (update_column_pointer[dim] + 1) \
                                             % self.dims[dim]


            mean_change, cov_change = self.check_stop_cond()
            if iteration != 0 and iteration % self.report == 0:
                self.report_metrics(iteration, start, mean_change, cov_change)

            if max(mean_change, cov_change) < self.epsilon:
                break

            iteration += 1
        return

    def update_hyper_parameter(self, dim, iteration):
        """
        :param dim:
        :return:
        """
        sigma = 0.0
        M = self.dims[dim]
        for j in range(M):
            m, S = self.posterior.get_vector_distribution(dim, j)
            sigma += np.trace(S) + np.dot(m, m)

        self.pSigma[dim] = sigma/(M*self.D)

    def update_natural_param_batch(self, dim, i, iteration):
        observed = self.tensor.find_observed_ui(dim, i)

        if len(observed) > self.batch_size:
            observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
            observed_subset = np.take(observed, observed_idx, axis=0)
        else:
            observed_subset = np.copy(observed)

        (m, S) = self.posterior.get_vector_distribution(dim, i)
        coords = np.array([entry[0] for entry in observed_subset])
        ys = np.array([entry[1] for entry in observed_subset])

        if self.likelihood_type == "normal":
            di, Di, si = self.estimate_di_Di_si_complete_conditional_batch(dim, i, coords, ys, m, S)
        else:
            di, Di, si = self.estimate_di_Di_si_batch(dim, i, coords, ys, m, S)

        scale = len(observed) / len(observed_subset)
        Di *= scale
        di *= scale

        # Compute next covariance and mean
        S_next   = self.update_cov_param(dim, i, m, S, di, Di)
        m_next   = self.update_mean_param(dim, i, m, S, di, Di)

        if self.noise_added:
            self.w_sigma = self.update_sigma_param(si, scale)

        # print("mean : ", np.linalg.norm(m_next - m), " cov: ", np.linalg.norm(S_next - S, "fro"))
        # Measures the change in the parameters from previous iterations
        self.keep_track_changes_params(dim, i, m, S, m_next, S_next)
        # Update the change
        self.posterior.update_vector_distribution(dim, i, m_next, S_next)

    def estimate_di_Di_si_batch(self, dim, i, coords, ys, m, S):
        num_subsamples     = np.size(coords, axis=0) # Number of subsamples

        othercols_left     = coords[:, : dim]
        othercols_right    = coords[:, dim + 1 :]
        othercols_concat   = np.concatenate((othercols_left, othercols_right), axis=1)

        alldims            = list(range(self.order))
        otherdims          = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        # Shape of vjs_batch would be (num_subsamples, k1, D)
        vjs_batch = self.sample_vjs_batch(othercols_concat, otherdims, self.k1)

        assert(num_subsamples == np.size(vjs_batch, axis=0)) # sanity check
        if self.noise_added:
            ws_batch   = np.random.rayleigh(np.square(self.w_sigma), size=(num_subsamples, self.k1))
        else:
            ws_batch   = None

        mean_batch = np.dot(vjs_batch, m) # Shape will be (num_samples, k1)
        cov_batch  = np.zeros((num_subsamples, self.k1)) # Shape will be (num_samples, k1)

        for num in range(num_subsamples):
            vs = vjs_batch[num, :, :] # shape (k1, D)
            cov_batch[num, :] = np.sum(np.multiply(vs.transpose(), np.inner(S, vs)), axis=0)

        di, Di, si = self.approximate_di_Di_si_with_second_layer_samplings(vjs_batch, ys, mean_batch, cov_batch, ws_batch)

        return di, Di, si

    @abstractmethod
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        raise NotImplementedError

    @abstractmethod
    def approximate_di_Di_si_with_second_layer_samplings(self, vjs_batch, ys, mean_batch, cov_batch, ws_batch):
        raise NotImplementedError

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
            # print("covGrad.norm: ", np.linalg.norm(covGrad, "fro"))
            covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
            # print("covStep: ", covStep)
            S_next = inv((1 - covStep) * inv(S) + np.multiply(covStep, covGrad))
        else:
            raise Exception("Unidentified update formula for covariance param")
        return S_next

    def update_sigma_param(self, si_acc, scale):
        # print("si: ", si_acc)
        # print("scale: ", scale)
        si_acc *= scale
        w_grad = -1/(2 * np.square(self.w_tau)) + si_acc
        # print("w_grad: ", w_grad)
        w_step = self.compute_stepsize_sigma_param(w_grad)
        # print("w_step ", w_step)
        update = (1-w_step) * (-0.5/np.square(self.w_sigma)) + w_step * w_grad
        # print("update: ", update)
        next_sigma = np.sqrt(-0.5/update)

        # print("sigma diff: ", next_sigma - self.w_sigma)
        return next_sigma

    def compute_stepsize_mean_param(self, dim, i, mGrad):
        # if self.scheme == "sgd":
        #     return self.eta/ (self.time_step[dim] + 1)

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
            return self.cov_eta/(self.time_step[dim] + 1)

        return self.cov_eta/(self.time_step[dim] + 1)

    def compute_stepsize_sigma_param(self, w_grad):
        self.w_ada_grad += np.square(w_grad)

        w_step = self.sigma_eta / self.w_ada_grad
        return w_step

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        if S.ndim != 1:
            cov_change  = np.linalg.norm(S_next - S, 'fro')
        else:
            cov_change  = np.linalg.norm(S_next - S)
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])

    def sample_vjs_batch(self, cols_batch, dims_batch, k):
        num_subsamples = np.size(cols_batch, axis=0)
        vjs_batch = np.ones((num_subsamples, k, self.D))

        for num in range(num_subsamples):
            cols = cols_batch[num, :]

            for dim, col in enumerate(cols):
                (mi, Si) = self.posterior.get_vector_distribution(dims_batch[dim], col)

                if Si.ndim == 1: #diagonal covariance
                    vj_sample = np.random.multivariate_normal(mi, np.diag(Si), size=k)
                else:
                    vj_sample = np.random.multivariate_normal(mi, Si, size=k)

                vjs_batch[num, :, :] = np.multiply(vjs_batch[num, :, :], vj_sample)

        return vjs_batch

    def sample_vjs(self, othercols, otherdims):
        uis = np.ones((self.D,))
        for dim, col in enumerate(othercols):
            # Sample from the approximate posterior
            (mi, Si) = self.posterior.get_vector_distribution(otherdims[dim], col)

            if Si.ndim == 1:
                uj_sample = np.random.multivariate_normal(mi, np.diag(Si))
            else:
                uj_sample = np.random.multivariate_normal(mi, Si)

            uis = np.multiply(uis, uj_sample)

        return uis

    """
    Functions to report metrics (error, vlb, etc)
    """
    def report_metrics(self, iteration, start, mean_change, cov_change):
        if iteration == self.report:
            print("Tensor dimensions: ", self.dims)
            print("Optimization metrics: ")
            print("Mean update scheme: ", self.mean_update)
            print("Covariance update : ", self.cov_update)
            print("k1 samples = ", self.k1, " k2 samples = ", self.k2)
            print("eta = ", self.eta, " cov eta = ", self.cov_eta, " sigma eta = ", self.sigma_eta)
            print("iteration |   time   | test_err | train_err|  d_mean  |   d_cov  |", end=" ")
            if self.likelihood_type == "poisson":
                print("test_nll | train_nll ")
            else:
                print("")
        current = time.time()
        dec = 4

        print('{:^10} {:^10} {:^10} {:^10} {:^10} {:^10}'\
              .format(iteration, np.around(current - start,2),\
              np.around(self.evaluate_test_error(), dec),\
              np.around(self.evaluate_train_error(), dec), \
              np.around(mean_change, dec),\
              np.around(cov_change, dec)), end=" ")

        if self.likelihood_type == "poisson":
            print('{:^10} {:^10}'\
                  .format(np.around(\
                            self.evaluate_nll(self.tensor.test_entries, self.tensor.test_vals), dec), \
                          np.around(\
                            self.evaluate_nll(self.tensor.train_entries, self.tensor.train_vals), dec)))
        else:
            print("")

    def estimate_vlb(self, entries, values):
        vlb = 0.0
        for i in range(len(values)):
            entry = entries[i]
            val   = values[i]
            m, S  = self.compute_posterior_param(entry)

    def estimate_expected_log_likelihood(self, m, S):
        raise NotImplementedError

    def compute_KL_divergence(self, m, S):
        raise NotImplementedError

    def evaluate_nll(self, entries, values):
        nll = 0.
        for i in range(len(values)):
            entry = entries[i]
            k     = values[i]
            m, S  = self.compute_posterior_param(entry)
            likelihood = self.compute_gauss_hermite(k, m, S)
            nll       -= np.log(likelihood)
        return nll

    def check_stop_cond(self):
        """
        :return: boolean
        Check for stopping condition
        """
        max_mean_change = 0
        max_cov_change  = 0
        for dim in range(len(self.dims)):
            for col in range(self.dims[dim]):
                mean_change, cov_change = self.norm_changes[dim][col, :]
                max_mean_change = max(max_mean_change, mean_change)
                max_cov_change  = max(max_cov_change, cov_change)

        return max_mean_change, max_cov_change

    def evaluate_train_error(self):
        return self.evaluate_error(self.tensor.train_entries, self.tensor.train_vals)

    def evaluate_test_error(self):
        return self.evaluate_error(self.tensor.test_entries, self.tensor.test_vals)

    def evaluate_error(self, entries, vals):
        """
        :return: error from a set of entries and associated correct values
        """
        error = 0.0
        for i in range(len(entries)):
            predict = self.predict_entry(entries[i])
            correct = vals[i]
            # print(predict, " vs ", correct)

            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            elif self.likelihood_type == "poisson":
                error += np.abs(predict - correct)
            else:
                return 0

        return error/len(entries)

    def predict_y_given_m(self, m):
        if self.likelihood_type == "normal":
            return m
        elif self.likelihood_type == "bernoulli":
            return 1 if m >= 0 else -1
        else:
            raise Exception("Unidentified likelihood type")

    def compute_fyi(self, k, yi, m, s):
        A  = self.link_fun(np.sqrt(2 * s) * yi + m)
        temp1 = np.power(A,k)/math.factorial(k)
        temp3 = np.exp(-A) * np.sqrt(2 * s)
        return temp1 * temp3

    def compute_gauss_hermite(self, k, m, S):
        res = 0.
        for i in range(len(self.hermite_points)):
            yi = self.hermite_points[i]
            weight = self.hermite_weights[i]
            res += weight * self.compute_fyi(k, yi, m, S)
        return np.divide(res, np.sqrt(2* np.pi * S))

    def compute_posterior_param(self, entry):
        ndim = len(entry)

        m    = np.ones((self.D, ))
        S    = np.ones((self.D, self.D))

        ms_acc = np.ones((self.D, ndim))
        all_ms = np.ones((self.D, ndim))
        Cs_acc = np.ones((self.D, self.D, ndim))
        all_Cs = np.ones((self.D, self.D, ndim))

        for dim, i in enumerate(entry):
            mi, Ci = self.posterior.get_vector_distribution(dim, i)
            m = np.multiply(m, mi)
            S = np.multiply(S, Ci)

            all_ms[:, dim] = mi
            all_Cs[:, :, dim] = Ci

            for d in range(ndim):
                if d != dim:
                    ms_acc[:, d] = np.multiply(ms_acc[:,d], mi)
                    Cs_acc[:, :, d] = np.multiply(Cs_acc[:, :, d], Ci)

        m = np.sum(m)
        S = np.sum(S)

        for d in range(ndim):
            S += np.dot(all_ms[:, d], np.inner(Cs_acc[:, :, d], all_ms[:, d]))
            S += np.dot(ms_acc[:, d], np.inner(all_Cs[:, :, d], ms_acc[:, d]))

        return m, S

    def compute_expected_count(self, m, S):
        num = self.max_count - self.min_count + 1
        probs = np.zeros((num, ))
        sum_probs = 0.

        k_array   = list(range(self.min_count, self.max_count + 1))

        for i, k in enumerate(k_array):
            prob = self.compute_gauss_hermite(k, m, S)
            probs[i]   = prob
            sum_probs += probs[i]

        return np.sum(np.multiply(probs, np.array(k_array)))/sum_probs

    def predict_entry(self, entry):
        # If not count-valued tensor
        if self.likelihood_type != "poisson":
            u = np.ones((self.D,))
            for dim, col in enumerate(entry):
                m, _ = self.posterior.get_vector_distribution(dim, col)
                u = np.multiply(u, m)
            m = np.sum(u)
            return self.predict_y_given_m(m)
        else:
            # if predicting count values, need to do estimation
            m, S = self.compute_posterior_param(entry)
            res  = self.compute_expected_count(m, S)
            return np.rint(res)