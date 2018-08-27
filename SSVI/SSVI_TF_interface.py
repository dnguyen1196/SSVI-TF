from abc import abstractclassmethod, abstractmethod
import Probability.ProbFun as probs
import numpy as np

from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import cholesky
from Model.TF_Models import Posterior_Full_Covariance, Posterior_Diag_Covariance

from Probability.normal import NormalDistribution
from Probability.bernoulli import BernoulliDistribution
from Probability.poisson import PoissonDistribution

import math
import time

class SSVI_TF(object):
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", noise_update="N", \
                        diag=False, mean0=None, cov0=None, sigma0=1,\
                        unstable_cov=False, k1=64, k2=64, batch_size=128, \
                        eta=1, cov_eta=1, sigma_eta=1, randstart=True):

        self.tensor = tensor
        self.dims   = tensor.dims
        self.datatype   = tensor.datatype
        self.order      = len(tensor.dims)   # number of dimensions
        self.randstart  = randstart
        self.D      = rank
        self.mean_update = mean_update
        self.cov_update  = cov_update
        self.noise_update = noise_update

        self.diag = diag
        if not diag:
            self.posterior        = Posterior_Full_Covariance(self.dims, self.D, mean0, cov0, randstart)
        else:
            self.posterior        = Posterior_Diag_Covariance(self.dims, self.D, mean0, cov0, randstart)
            self.ada_acc_grad_cov = [np.zeros((self.D, s)) for s in self.dims]

        self.mean0    = mean0
        self.cov0     = cov0
        self.w_sigma0 = sigma0
        self.unstable_cov = unstable_cov
        self.cov_epsilon  = 1e-7
        self.min_eigenval = 0.01

        self.k1     = k1
        self.k2     = k2
        self.batch_size = batch_size
        self.predict_num_samples = 32

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
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.dims]

        self.eta = eta
        self.cov_eta = cov_eta
        self.poisson_eta = 0.1
        self.sigma_eta = sigma_eta

        # keep track of changes in norm
        self.norm_changes = [np.zeros((s, 2)) for s in self.dims]
        self.noise_added  = False

        self.d_mean = 1.
        self.d_cov  = 1.

    def factorize(self, report=100, max_iteration=2000, fixed_covariance=False, to_report=[], detailed_report=True):
        self.report = report
        self.header_printed = False
        self.to_report = to_report
        self.max_iteration = max_iteration
        self.fixed_covariance = fixed_covariance
        self.detailed_report = detailed_report
        self.missing = {}
        self.max_changes = [[0, 0] for _ in self.dims]

        print("Factorizing with max_iteration =", max_iteration, " fixing covariance?: ", fixed_covariance)

        update_column_pointer = [0 for _ in range(self.order)]
        start = time.time()

        for iteration in range(self.max_iteration+1):
            current = time.time()

            for dim in range(self.order):
                col = update_column_pointer[dim]
                # Update the natural params of the col-th factor
                # in the dim-th dimension
                # self.update_natural_params(dim, col, iteration)
                self.update_natural_param_batch(dim, col)
                self.update_hyper_parameter(dim)

            # Move on to the next column of the hidden matrices
            for dim in range(self.order):
                if (update_column_pointer[dim] + 1 == self.dims[dim]):
                    self.time_step[dim] += 1  # increase time step
                    #self.max_changes[dim] = [0,0] # Reset max_changes

                update_column_pointer[dim] = (update_column_pointer[dim] + 1) \
                                             % self.dims[dim]

            mean_change, cov_change = self.check_stop_cond()
            #mean_change = max(x[0] for x in self.max_changes)
            #cov_change  = max(x[1] for x  in self.max_changes)
            #print(mean_change, cov_change)

            if (iteration != 0 and (iteration % self.report == 0)) or iteration in to_report:
                if self.detailed_report:
                    self.report_metrics(iteration, start, mean_change, cov_change)
                else:
                    self.report_metrics_mini(iteration, start, mean_change, cov_change)

            if max(mean_change, cov_change) < self.epsilon:
                break

    def update_hyper_parameter(self, dim):
        """
        :param dim:
        :return:
        """
        sigma = 0.0
        M = self.dims[dim]
        for j in range(M):
            m, S = self.posterior.get_vector_distribution(dim, j)
            if self.diag:
                sigma += np.sum(S) + np.dot(m, m)
            else:
                sigma += np.trace(S) + np.dot(m, m)

        self.pSigma[dim] = sigma/(M*self.D)

    def update_natural_param_batch(self, dim, i):
        observed = self.tensor.find_observed_ui(dim, i)

        if len(observed) == 0: # If there is no data associated with the column
            self.missing[(dim, i)] = True
            return

        if len(observed) > self.batch_size:
            observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
            observed_subset = np.take(observed, observed_idx, axis=0)
        else:
            observed_subset = observed

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

        assert(not np.any(np.iscomplex(di)))
        assert(not np.any(np.iscomplex(Di)))
        assert(not np.any(np.isnan(Di)))
        assert(not np.any(np.isinf(Di)))

        # Compute next covariance and mean
        if self.diag:
            S_next   = self.update_cov_param_diag(dim, i, m, S, di, Di)
            m_next   = self.update_mean_param_diag(dim, i, m, S, di, Di)
        else:
            S_next   = self.update_cov_param(dim, i, m, S, di, Di)
            m_next   = self.update_mean_param(dim, i, m, S, di, Di)

        # Sanity check
        assert(not np.any(np.iscomplex(S_next)))
        assert(not np.any(np.isnan(S_next)))
        assert(not np.any(np.isinf(S_next)))
        #assert(not np.any(np.iscomplex(m_next)))

        # if self.unstable_cov:
        #     add    = np.full((self.D,), self.cov_epsilon)
        #     S_next = S_next + np.diag(add) if not self.diag else S_next + add

        if self.noise_added:
            w_sigma        = self.update_sigma_param(si, scale)
            self.w_changes = np.abs(w_sigma - self.w_sigma)
            self.w_sigma   = w_sigma

        # If chosen fixed covariance, skip updating covariance (test purpose)
        if self.fixed_covariance:
            S_next = S

        # Measures the change in the parameters from previous iterations
        self.keep_track_changes_params(dim, i, m, S, m_next, S_next)
        # Update the change
        self.posterior.update_vector_distribution(dim, i, m_next, S_next)

    def estimate_di_Di_si_batch(self, dim, i, coords, ys, m, S):
        """
        :param dim:
        :param i:
        :param coords:
        :param ys:
        :param m:
        :param S:
        :return:

        Note that robust model will have a completely different implementation of this
        function
        """
        num_subsamples     = np.size(coords, axis=0) # Number of subsamples

        othercols_left     = coords[:, : dim]
        othercols_right    = coords[:, dim + 1 :]
        othercols_concat   = np.concatenate((othercols_left, othercols_right), axis=1)

        alldims            = list(range(self.order))
        otherdims          = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        # Shape of vjs_batch would be (num_subsamples, k1, D)
        # Sample vj, tk, ...
        vjs_batch = self.sample_vjs_batch(othercols_concat, otherdims, self.k1)

        assert(num_subsamples == np.size(vjs_batch, axis=0)) # sanity check

        # Sample rayleigh noise
        if self.noise_added:
            ws_batch   = np.random.rayleigh(np.square(self.w_sigma), size=(num_subsamples, self.k1))
        else:
            ws_batch   = None

        mean_batch = np.dot(vjs_batch, m) # Shape will be (num_samples, k1)
        var_batch  = np.zeros((num_subsamples, self.k1)) # Shape will be (num_samples, k1)

        for num in range(num_subsamples):
            vs = vjs_batch[num, :, :] # shape (k1, D)
            if self.diag:
                var_batch[num, :] = np.sum(np.multiply(vs.transpose(), np.inner(np.diag(S), vs)), axis=0)
            else:
                var_batch[num, :] = np.sum(np.multiply(vs.transpose(), np.inner(S, vs)), axis=0)

        di, Di, si = self.approximate_di_Di_si_with_second_layer_samplings(vjs_batch, ys, mean_batch, var_batch, ws_batch)

        return di, Di, si

    @abstractmethod
    def estimate_di_Di_si_complete_conditional_batch(self, dim, i, coords, ys, m, S):
        raise NotImplementedError

    @abstractmethod
    def approximate_di_Di_si_with_second_layer_samplings(self, vjs_batch, ys, mean_batch, cov_batch, ws_batch):
        raise NotImplementedError

    def update_mean_param(self, dim, i, m, S, di_acc, Di_acc):
        sigma = self.pSigma[dim]
        pSigma_inv = np.diag(np.full((self.D,), 1/sigma))

        if self.mean_update == "S":
            meanGrad = (np.inner(pSigma_inv, self.pmu - m) + di_acc)
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            m_next = m + np.multiply(meanStep, meanGrad)

        elif self.mean_update == "N":
            C = np.inner(inv(S), m)
            meanGrad = np.inner(pSigma_inv, self.pmu) + di_acc - 2 * np.inner(Di_acc, m)
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            C_next   = np.multiply(1 -  meanStep, C) + meanStep * meanGrad
            m_next   = np.inner(S, C_next)

        else:
            raise Exception("Unidentified update formula for covariance param")
        return m_next

    def update_cov_param(self, dim, i, m, S, di_acc,  Di_acc):
        sigma = self.pSigma[dim]
        pSigma_inv = np.diag(np.full((self.D,), 1/sigma))

        if self.cov_update == "S":
            L = cholesky(S)
            covGrad = np.triu(inv(np.multiply(L, np.eye(self.D))) \
                      - np.inner(L, pSigma_inv) + 2 * np.dot(L, Di_acc))

            covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
            L_next  = L + covStep * covGrad
            S_next  = np.dot(L, np.transpose(L))

        elif self.cov_update == "N":
            covGrad = (pSigma_inv - 2 * Di_acc)
            covStep = self.compute_stepsize_cov_param(dim, i, covGrad)
            S_next = inv((1 - covStep) * inv(S) + np.multiply(covStep, covGrad))

        else:
            raise Exception("Unidentified update formula for covariance param")

        #S_next = S_next + np.eye(self.D) * self.cov_eta
        w,v = np.linalg.eigh(S_next)
        w   = np.maximum(np.real(w), self.min_eigenval)
        S_next = np.dot(v, np.dot(np.diag(w),v.transpose()))
        return S_next

    def update_mean_param_diag(self, dim, i, m, S, di_acc, Di_acc):
        sigma = self.pSigma[dim]
        pSigma_inv = np.full((self.D,), 1/sigma)

        if self.mean_update == "S":
            meanGrad = np.multiply(pSigma_inv, self.pmu - m) + di_acc
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            m_next = m + np.multiply(meanStep, meanGrad)

        elif self.mean_update == "N":
            C = np.multiply(np.reciprocal(S), m)
            meanGrad = np.multiply(pSigma_inv, self.pmu) + di_acc - 2 * np.multiply(Di_acc, m)
            meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
            C_next   = np.multiply(1 -  meanStep, C) + meanStep * meanGrad
            m_next   = np.multiply(S, C_next) # S S^(-1) m
        else:
            raise Exception("Unidentified update formula for covariance param")

        return m_next

    def update_cov_param_diag(self, dim, i, m, S, di_acc,  Di_acc):
        sigma = self.pSigma[dim]
        pSigma_inv = np.full((self.D,), 1/sigma)

        if self.cov_update == "S":
            L = np.sqrt(S)
            # covGrad.shape = (D,)
            covGrad = np.reciprocal(L) - np.multiply(L, pSigma_inv) + 2 * np.multiply(L, Di_acc)

            covStep = self.compute_step_size_cov_param_diag(dim, i, covGrad)
            L_next  = L + covStep * covGrad
            S_next  = np.square(L)

        elif self.cov_update == "N":
            covGrad = pSigma_inv - 2 * Di_acc
            covStep = self.compute_step_size_cov_param_diag(dim, i, covGrad)
            S_next = np.reciprocal((1 - covStep) * np.reciprocal(S) + np.multiply(covStep, covGrad))

        else:
            raise Exception("Unidentified update formula for covariance param")
        S_next = np.maximum(S_next, self.epsilon)
        return S_next

    def update_sigma_param(self, si_acc, scale):
        si_acc *= scale
        w_grad = -1/(2 * np.square(self.w_tau)) + si_acc
        w_step = self.compute_stepsize_sigma_param(w_grad)

        update = (1-w_step) * (-0.5/np.square(self.w_sigma)) + w_step * w_grad
        next_sigma = np.sqrt(-0.5/update)
        return next_sigma

    def compute_stepsize_mean_param(self, dim, i, mGrad):
        acc_grad = self.ada_acc_grad[dim][:, i]
        grad_sqr = np.square(mGrad)
        self.ada_acc_grad[dim][:, i] += grad_sqr

        return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        #if self.likelihood_type != "poisson":
        #    return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        #else:
        #    return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def compute_stepsize_cov_param(self, dim, i, covGrad):
        #if self.likelihood_type == "poisson":
        #    return self.cov_eta/(self.time_step[dim] + 1)

        return self.cov_eta/(self.time_step[dim] + 1)

    def compute_step_size_cov_param_diag(self, dim, i, covGrad):
        acc_grad = self.ada_acc_grad_cov[dim][:, i]
        grad_sqr = np.square(covGrad)
        self.ada_acc_grad_cov[dim][:, i] += grad_sqr

        if self.likelihood_type != "poisson":
            return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        else:
            return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def compute_stepsize_sigma_param(self, w_grad):
        self.w_ada_grad += np.square(w_grad)
        w_step = self.sigma_eta / self.w_ada_grad
        return w_step

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        # Check for dimensionality of covariance
        if S.ndim != 1:
            cov_change  = np.linalg.norm(S_next - S, 'fro')
        else:
            cov_change  = np.linalg.norm(S_next - S)
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])
        #d_mean = self.max_changes[dim][0]
        #d_cov = self.max_changes[dim][1]
        #self.max_changes[dim] = [max(d_mean, mean_change), max(d_cov, cov_change)]

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

                # print (vj_sample.shape)
                vjs_batch[num, :, :] = np.multiply(vjs_batch[num, :, :], vj_sample)

        return vjs_batch

    """
    Functions to report metrics (error, vlb, etc)
    """
    def report_metrics(self, iteration, start, mean_change, cov_change):
        if not self.header_printed and (iteration == self.report or (len(self.to_report) > 0 and iteration == self.to_report[0])):
            self.header_printed = True
            print("Tensor dimensions: ", self.dims)
            print("Optimization metrics: ")
            if hasattr(self, "window_size"):
                print("Using Ada Delta with window size = ", self.window_size)
            else:
                print("Using Ada Grad")
            print("Mean update scheme: ", self.mean_update)
            print("Covariance update : ", self.cov_update)
            print("Using diagonal covariance?: ", self.diag)
            print("Random start?", self.randstart)
            print("k1 samples = ", self.k1, " k2 samples = ", self.k2)
            print("eta = ", self.eta, " cov eta = ", self.cov_eta, " sigma eta = ", self.sigma_eta)
            print("iteration |   time   |test_rsme |rel-te-err|train_rsme|rel-tr-err|  d_mean  |   d_cov  |", end=" ")
            print("test_nll | train_nll |    VLB   |  E-term  |    KL    ", end=" ")

            if self.noise_added:
                print("|   dw   |")
            else:
                print("")

        current = time.time()
        dec = 4
        rsme_train, error_train = self.evaluate_train_error()
        rsme_test, error_test   = self.evaluate_test_error()
        E_term, KL = self.estimate_vlb(self.tensor.train_entries, self.tensor.train_vals)

        print('{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}'\
              .format(iteration, np.around(current - start,2),\
              np.around(rsme_test, dec), \
              np.around(error_test, dec), \
              np.around(rsme_train, dec), \
              np.around(error_train, dec),\
              np.around(mean_change, dec),\
              np.around(cov_change, dec)), end=" ")

        test_nll = self.estimate_negative_log_likelihood(self.tensor.test_entries, self.tensor.test_vals)
        train_nll = self.estimate_negative_log_likelihood(self.tensor.train_entries, self.tensor.train_vals)

        print('{:^10} {:^10} {:^10} {:^10} {:^10}'\
              .format(np.around(test_nll, 2), \
                      np.around(train_nll, 2),\
                      np.around(E_term - KL),\
                      np.around(E_term, 3),\
                      np.around(KL, 3)), end=" ")

        if self.noise_added:
            print('{:^10}'.format(np.around(self.w_changes, dec)))
        else:
            print("")

    def report_metrics_mini(self, iteration, start, mean_change, cov_change):
        if not self.header_printed and (iteration == self.report or (len(self.to_report) > 0 and iteration == self.to_report[0])):
            self.header_printed = True
            print("Tensor dimension:", self.dims)
            print("Mean update:", self.mean_update)
            print("Cov update:", self.cov_update)
            print("Using diagonal covariance:", self.diag)
            print("k1 samples = ", self.k1, " k2 samples = ", self.k2)
            print("eta = ", self.eta, " cov eta = ", self.cov_eta, " sigma eta = ", self.sigma_eta)

            #print("iteration |   time   |test_rsme |train_rsme|  d_mean  |   d_cov  |")

            print("iteration |   time   |test_rsme|  d_mean  |   d_cov  |")

        current = time.time()
        dec = 4

            #rsme_train, error_train = self.evaluate_train_error()
        rsme_test, error_test   = self.evaluate_test_error()

            #print("{:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(iteration, np.around(current - start,2), rsme_test, rsme_train, np.around(mean_change, dec), np.around(cov_change, dec)))
        print("{:^10} {:^10} {:^10} {:^10} {:^10}".format(iteration, np.around(current-start, 2), np.around(rsme_test,dec), np.around(mean_change, dec), np.around(cov_change, dec)))

    def estimate_vlb(self, entries, values):
        """
        """
        vlb = 0.
        KL  = self.estimate_KL_divergence()
        for i, entry in enumerate(entries):
            vlb += self.estimate_expectation_term_vlb(entry, values[i])
        return vlb, KL

    def estimate_expectation_term_vlb(self, entry, val):
        """
        Estimate the expectation term in the VLB
        by samplings
        """
        k = 20
        sampled_vectors_prod = np.ones((k, self.D))

        for dim, col in enumerate(entry):
            m, S = self.posterior.get_vector_distribution(dim, col)
            if self.diag:
                samples = np.random.multivariate_normal(m, np.diag(S), size=k)
            else:
                samples = np.random.multivariate_normal(m, S, size=k)
            sampled_vectors_prod *= samples

        ms = np.sum(sampled_vectors_prod, axis=1) # shape = (k,)
        if self.noise_added:
            ws = np.random.rayleigh(np.square(self.w_sigma), (k,))
            fs = np.random.normal(ms, ws, size=(k, k))
        else:
            fs = ms
        s = self.likelihood_param
        expected_ll = np.mean(self.likelihood.log_pdf(val, self.link_fun(fs), s))
        return expected_ll

    def estimate_KL_divergence(self):
        """
        Estimate the KL divergence components of the VLB
        """
        KL = 0.
        for dim, ncol in enumerate(self.dims):
            for col in range(ncol):
                m, S = self.posterior.get_vector_distribution(dim, col)
                sigma = self.pSigma[dim]
                S1    = np.diag(np.full((self.D,),sigma))
                m1    = self.pmu
                if self.diag:
                    KL = self.KL_distance_multivariate_normal(m, np.diag(S), m1, S1)
                else:
                    KL = self.KL_distance_multivariate_normal(m, S, m1, S1)
        return KL

    def KL_distance_multivariate_normal(self, m0, S0, m1, S1):
        KL = 0.5 * (np.trace(np.dot(inv(S1),S0)) \
                        + np.dot(np.transpose(m1 - m0), np.dot(inv(S1), m1 - m0))\
                        - self.D + np.log(np.linalg.det(S1)/ np.linalg.det(S0)))
        return KL

    def estimate_negative_log_likelihood(self, entries, vals):
        nll = 0.
        dims = np.arange(len(self.dims))
        s    = self.likelihood_param

        for i in range(len(vals)):
            entry = entries[i]
            expanded_entry = np.expand_dims(entry, axis=0)
            vjs = self.sample_vjs_batch(expanded_entry, dims, self.k1) # vjs.shape = (k1, D)
            # print("vjs.shape = ", vjs.shape)
            vs = vjs[0, :, :]

            ms  = np.sum(vs, axis=1) # shape (k1,)
            if self.noise_added:
                w   = np.random.rayleigh(np.square(self.w_sigma), self.k1)
                fs  = np.random.normal(np.full_like(w,ms), w, size=(self.k1, self.k1))
            else:
                fs  = ms

            # fs  = np.random.normal(ms, w, size=(self.k1, self.k1))
            # expected_ll = np.mean(np.mean(self.likelihood.pdf(vals[i], fs, s), axis=0))
            expected_ll = np.mean(self.likelihood.pdf(vals[i], fs, s))
            nll -= np.log(expected_ll)

        return nll

    def check_stop_cond(self):
        """
        :return: boolean
        Check for stopping condition
        """
        d_mean = 0
        d_cov  = 0

        for dim in range(len(self.dims)):
            for col in range(self.dims[dim]):
                mean_change, cov_change = self.norm_changes[dim][col, :]
                d_mean = max(d_mean, mean_change)
                d_cov  = max(d_cov, cov_change)

        self.d_mean = d_mean
        self.d_cov  = d_cov

        return d_mean, d_cov

    def evaluate_train_error(self):
        # error = self.evaluate_error(self.tensor.train_entries, self.tensor.train_vals)
        rsme, error = self.evaluate_RSME(self.tensor.train_entries, self.tensor.train_vals)
        return rsme, error

    def evaluate_test_error(self):
        # error = self.evaluate_error(self.tensor.test_entries, self.tensor.test_vals)
        rsme, error = self.evaluate_RSME(self.tensor.test_entries, self.tensor.test_vals)
        return rsme, error

    def evaluate_error(self, entries, vals):
        """
        :return: error from a set of entries and associated correct values
        """
        error = 0.0
        for i in range(len(entries)):
            predict = self.predict_entry(entries[i])
            correct = vals[i]
            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            elif self.likelihood_type == "poisson":
                error += np.abs(predict - correct)
            else:
                return 0

        return error/len(entries)

    def evaluate_RSME(self, entries, vals):
        """
        TODO: Have option to do average?
        """

        rsme = 0.0
        error = 0.0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            missing_col = False

            for dim, col in enumerate(entry):
                # If the entry involve an unlearned column, flag it
                if (dim, col) in self.missing:
                    num_entries -= 1
                    missing_col = True
                    break

            if missing_col:
                continue # Skip entries that involve unlearned column

            predict = self.predict_entry(entry)
            correct = vals[i]
            rsme += np.square(predict - correct)

            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            elif self.likelihood_type == "poisson":
                error += np.abs(predict - correct)

        rsme = np.sqrt(rsme/num_entries)
        error = error/num_entries
        return rsme, error

    """
    Predict entry function
    """

    def predict_entry(self, entry):
        # real value data for all models
        if self.likelihood_type == "normal":
            u = np.ones((self.D,))
            for dim, col in enumerate(entry):
                m, _ = self.posterior.get_vector_distribution(dim, col)
                u = np.multiply(u, m)
            return np.sum(u)

        # deterministic binary
        elif self.likelihood_type == "bernoulli" and not self.noise_added:
            u = np.ones((self.D,))
            for dim, col in enumerate(entry):
                m, _ = self.posterior.get_vector_distribution(dim, col)
                u = np.multiply(u, m)
            return 1 if np.sum(u) > 0 else -1
        # For other cases, do samplings to estimate
        else:
            res = self.estimate_expected_observation_via_sampling(entry)
            if self.likelihood_type == "bernoulli":
                return 1 if res >= 1/2 else -1
            elif self.likelihood_type == "poisson":
                return np.rint(res)
    """
    Bridge function
    """
    def predict_y_given_m(self, m):
        if self.likelihood_type == "normal":
            return m
        elif self.likelihood_type == "bernoulli":
            return 1 if m >= 0 else -1
        else:
            raise Exception("Unidentified likelihood type")

    """
    Functions to do count value prediction through the use of gauss-hermite
    quadrature
    """
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

    def compute_expected_count_quadrature(self, m, S):
        num = self.max_count - self.min_count + 1
        probs = np.zeros((num, ))
        sum_probs = 0.

        k_array   = list(range(self.min_count, self.max_count + 1))

        for i, k in enumerate(k_array):
            prob = self.compute_gauss_hermite(k, m, S)
            probs[i]   = prob
            sum_probs += probs[i]

        return np.sum(np.multiply(probs, np.array(k_array)))/sum_probs

    """
    Functions to do prediction via samplings
    """
    def compute_expected_count_sampling(self, entry):
        ms = np.ones((self.predict_num_samples, self.D))

        for dim, i in enumerate(entry):
            m, S = self.posterior.get_vector_distribution(dim, i)
            samples = np.random.multivariate_normal(m, S, size=(self.predict_num_samples))
            ms = ms * samples

        ms = np.sum(ms, axis=1) # fs.shape == (self.predict_num_samples, )

        # TODO: check dimensionality
        if self.noise_added:
            ws = np.random.rayleigh(self.w_sigma**2, size=self.predict_num_samples)
            fs = np.random.normal(ms, ws, size=(self.predict_num_samples))
            fs = self.link_fun(fs)
            counts = self.likelihood.sample(fs, (self.predict_num_samples))
        else:
            fs = ms
            fs = self.link_fun(fs)
            counts = self.likelihood.sample(fs, (self.predict_num_samples))

        return np.mean(counts)

    def estimate_expected_observation_via_sampling(self, entry):
        ms = np.ones((self.predict_num_samples, self.D))

        for dim, i in enumerate(entry):
            m, S = self.posterior.get_vector_distribution(dim, i)
            if self.diag:
                samples = np.random.multivariate_normal(m, np.diag(S), size=(self.predict_num_samples))
            else:
                samples = np.random.multivariate_normal(m, S, size=(self.predict_num_samples))
            ms = ms * samples

        ms = np.sum(ms, axis=1) # fs.shape == (self.predict_num_samples, )

        if self.noise_added:
            ws = np.random.rayleigh(np.square(self.w_sigma), size=(self.predict_num_samples,))
            fs = np.random.normal(ms, ws, size=(self.predict_num_samples, self.predict_num_samples))
        else: # deterministic model
            fs = ms

        predict = self.link_fun(fs)
        return np.mean(predict)

    """
    Reset function
    """
    def reset(self):
        self.posterior.reset()
        self.w_sigma         = self.w_sigma0
        self.pSigma          = np.ones((len(self.dims),))

        self.time_step = [0 for _ in range(self.order)]

        # Optimization parameter
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.dims]
        self.norm_changes = [np.zeros((s, 2)) for s in self.dims]

        if self.noise_added:
            # For simple and robust model
            self.w_tau = 1.
            self.w_sigma = 1.
            self.w_ada_grad = 0.

    def evaluate_true_params(self):
        train_rsme, train_error = self.evaluate_true_hidden_vectors(self.tensor.train_entries, \
                                                                    self.tensor.train_vals, \
                                                                    self.tensor.matrices)
        test_rsme, test_error = self.evaluate_true_hidden_vectors(self.tensor.test_entries, \
                                                                    self.tensor.test_vals, \
                                                                    self.tensor.matrices)

        test_nll = self.evaluate_true_model_nll(self.tensor.test_entries,\
                                                self.tensor.test_vals,\
                                                self.tensor.matrices)

        train_nll = self.evaluate_true_model_nll(self.tensor.train_entries,\
                                                 self.tensor.train_vals,\
                                                 self.tensor.matrices)

        print("Evaluation for true params: ")
        print(" test_rsme | train_rsme | rel-te-err | rel-tr-err |  test_nll  |  train_nll |")
        print("{:^12} {:^12} {:^12} {:^12} {:^12} {:^12}".format(\
            np.around(test_rsme, 10), \
            np.around(train_rsme, 10), \
            np.around(test_error, 10), \
            np.around(train_error, 10), \
            np.around(test_nll, 2), \
            np.around(train_nll, 2)))

    def evaluate_true_hidden_vectors(self, entries, vals, matrices):
        rsme = 0.0
        error = 0.0

        for i in range(len(entries)):
            predict = self.true_model_predict(entries[i], matrices)
            correct = vals[i]
            rsme += np.square(predict - correct)

            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            elif self.likelihood_type == "poisson":
                error += np.abs(predict - correct)

        rsme = np.sqrt(rsme/len(vals))
        error = error/len(vals)
        return rsme, error

    # TODO: what to do with noise? for now just ignore it?
    def true_model_predict(self, entry, matrices):
        actual_d = self.tensor.D
        ms       = np.ones((actual_d,))
        for dim, col in enumerate(entry):
            ms = np.multiply(ms, matrices[dim][col, :])

        f  = np.sum(ms)
        if self.datatype == "real":
            return f
        elif self.datatype == "binary":
            # return 1 if f > 1/2 else -1
            return 1 if self.true_model_predict_via_sampling(f) > 1/2 else -1
        elif self.datatype == "count":
            return self.true_model_predict_via_sampling(f)


    def true_model_predict_via_sampling(self, f):
        noise_ratio = self.tensor.noise_ratio
        if noise_ratio:
            w = np.abs(f) * noise_ratio
        else:
            w = self.tensor.noise

        num_samples = 50
        ws = np.random.normal(0, w, size=(num_samples,))
        f_noised = np.add(f, ws)
        ps = self.link_fun(f_noised)
        p = np.mean(ps)
        #return p
        return np.rint(p)

    def evaluate_true_model_nll(self, entries, vals, matrices):
        nll  = 0.
        dims = np.arange(len(self.dims))
        s    = self.likelihood_param

        for i in range(len(vals)):
            ms = np.ones((self.tensor.D,))
            for dim, col in enumerate(dims):
                ms = np.multiply(ms, self.tensor.matrices[dim][col, :])

            f = np.sum(ms)

            if self.noise_added:
                noise_ratio = self.tensor.noise_ratio
                if noise_ratio:
                    w = np.abs(f) * noise_ratio
                else:
                    w = self.tensor.noise

                fs  = np.random.normal(f, w, size=(self.k1,))
            else:
                fs = f

            expected_ll = np.mean(self.likelihood.pdf(vals[i], fs, s))
            nll -= np.log(expected_ll)

        return nll
