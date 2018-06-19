import Probability.ProbFun as probs
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import cholesky
from Model.TF_Models import ApproximatePosteriorParams
from Probability.normal import NormalDistribution
from Probability.bernoulli import BernoulliDistribution
from Probability.poisson import PoissonDistribution

import math
from math import log10, floor
import time

"""
SSVI_TF.py
SSVI algorithm to learn the hidden matrix factors behind tensor
factorization
"""
class SSVI_TF_robust():
    def __init__(self, tensor, rank, mean_update="S", cov_update="N", mean0=None, cov0= None, k1=10, k2=10, batch_size=100):
        self.dims       = tensor.dims        # dimension of the tensors
        self.order      = len(tensor.dims)   # number of dimensions
        self.tensor     = tensor
        self.D          = rank
        self.datatype   = tensor.datatype

        self.posterior  = ApproximatePosteriorParams(self.dims, self.D, mean0, cov0)

        self.mean_update  = mean_update
        self.cov_update   = cov_update

        # Get prior mean and covariance
        self.pmu             = np.ones((self.D,))
        self.pSigma          = np.ones((len(self.dims),))
        self.pSigma_inv      = np.eye(self.D)

        self.w_tau = 1.
        self.w_sigma = 1.
        self.w_ada_grad = 0.

        if self.datatype == "real":
            self.link_fun = lambda m : m
            self.likelihood_type = "normal"
            self.likelihood_param = 1
            self.probs = NormalDistribution()
        elif self.datatype == "binary":
            self.link_fun = lambda m : probs.sigmoid(m)
            self.likelihood_type = "bernoulli"
            self.likelihood_param = 0 # Not used
            self.probs = BernoulliDistribution()

        elif self.datatype == "count":
            self.likelihood_type = "poisson"
            self.max_count   = tensor.max_count
            self.min_count   = tensor.min_count
            self.herm_degree = 50
            self.hermite_points, self.hermite_weights = np.polynomial.hermite.hermgauss(self.herm_degree)
            self.link_fun = lambda m : probs.poisson_link(m)
            self.likelihood_param = 0 # Not used
            self.probs = PoissonDistribution()

        # optimization scheme
        self.time_step = [1 for _ in range(self.order)]

        # Stochastic optimization parameters
        self.batch_size  = batch_size
        self.iterations  = 60001
        self.k1          = k1
        self.k2          = k2
        self.epsilon     = 0.0001

        # adagrad parameters
        self.offset       = 0.000001
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.dims]
        self.ada_acc_cov  = [np.zeros((self.D, self.D, s)) for s in self.dims]
        self.eta          = 1
        self.poisson_eta  = 0.1

        # keep track of changes in norm
        self.norm_changes = [np.ones((s, 2)) for s in self.dims]

    def factorize(self, report=1000):
        self.report = report
        """
        factorize
        :return: None
        Doing round robin updates for each column of the
        hidden matrices
        """
        update_column_pointer = [0] * self.order
        start = time.time()
        iteration = 0
        while True:
            mean_change, cov_change = self.check_stop_cond()

            current = time.time()
            if iteration != 0 and iteration % self.report == 0:
                self.report_metrics(iteration, start, mean_change, cov_change)

            if max(mean_change, cov_change) < self.epsilon:
                break


            for dim in range(self.order):
                col = update_column_pointer[dim]
                # Update the natural params of the col-th factor
                # in the dim-th dimension
                self.update_natural_params(dim, col, iteration)
                self.update_hyper_parameter(dim, iteration)

            # Move on to the next column of the hidden matrices
            for dim in range(self.order):
                if (update_column_pointer[dim] + 1 == self.dims[dim]):
                    self.time_step[dim] += 1  # increase time step
                update_column_pointer[dim] = (update_column_pointer[dim] + 1) \
                                             % self.dims[dim]

            iteration += 1

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

        Di = np.zeros((self.D, self.D))
        di = np.zeros((self.D,))
        si = 0.

        for entry in observed_i:
            coord = entry[0]
            y = entry[1]

            # TODO: closed form expression for the complete conditional case
            # if self.likelihood_type == "normal": # Case of complete conditional
            #     (di_acc_update, Di_acc_update) = self.estimate_di_Di_complete_conditional(dim, i, coord, y, m, S)
            # else:
            (di_hat, Di_hat, si_hat) = self.estimate_di_Di_si(dim, i, coord, y, m, S)

            Di += Di_hat
            di += di_hat
            si += si_hat

        scale = len(observed_i) / min(self.batch_size, len(observed_i))
        Di *= scale
        di *= scale

        si *= scale

        # Compute next covariance and mean
        S_next   = self.update_cov_param(dim, i, m, S, di, Di)
        m_next   = self.update_mean_param(dim, i, m, S, di, Di)

        self.update_sigma_param(si)

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

    def update_sigma_param(self, si_acc):
        w_grad = -1/(2 * np.square(self.w_tau)) + 1/ (4 * np.square(self.w_sigma)) * si_acc
        w_step = self.compute_stepsize_sigma_param(w_grad)
        update = (1-w_step) * (-1/np.square(self.w_sigma)) + w_step * w_grad
        self.w_sigma = np.sqrt(-1/update)

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

    def compute_stepsize_sigma_param(self, w_grad):
        self.w_ada_grad += np.square(w_grad)
        w_step = self.eta / self.w_ada_grad
        return w_step

    def keep_track_changes_params(self, dim, i, m, S, m_next, S_next):
        mean_change = np.linalg.norm(m_next - m)
        cov_change  = np.linalg.norm(S_next - S, 'fro')
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])

    def estimate_di_Di_complete_conditional(self, dim, i, coord, y, mui, Sui):
        """
        :param dim:
        :param i:
        :param coord:
        :param y:
        :param m:
        :param S:
        :return:

        dij and Dij with closed form update for when the likelihood
        is Gaussian
        """
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
            ui = self.sample_uis(othercols, otherdims)
            w  = np.random.rayleigh(np.square(self.w_sigma))

            meanf     = np.dot(ui, m)
            covS     = np.dot(ui, np.inner(S, ui)) + w
            pdf, pdf_1st, pdf_2nd = self.approximate_expected_derivative_pdf(y, meanf, covS)

            delta_phi = ui * pdf_1st
            delta_squared_phi = np.outer(ui, ui) * pdf_2nd
            delta_log_phi = 1/pdf * delta_phi
            delta_squared_log_phi = 1/ pdf * (delta_squared_phi - 1/ pdf * np.outer(delta_phi, delta_phi))

            di += 1/pdf * delta_phi
            Di += 1/2 * delta_squared_log_phi
            si += w/(8 * np.square(self.w_sigma)) * pdf_2nd/ self.k1 * 1/pdf

        return di, Di, si

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

    def approximate_expected_derivative_pdf(self, y, meanf, varf):
        fs = np.random.normal(meanf, varf, (self.k2,))
        pdf = 0.
        pdf_prime = 0.
        pdf_double_prime = 0.
        s = self.likelihood_param
        for fij in fs:
            pdf += 1/self.k2 * self.probs.pdf(y, fij, s)
            pdf_prime += 1/self.k2 * self.probs.fst_derivative_pdf(y, fij, s)
            pdf_double_prime += 1/self.k2 * self.probs.snd_derivative_pdf(y, fij, s)
        return pdf, pdf_prime, pdf_double_prime

    def estimate_expected_derivative(self, y, meanf, covS) :
        first_derivative = 0.0
        snd_derivative = 0.0
        s = self.likelihood_param

        for k2 in range(self.k2):
            f = probs.sample("normal", (meanf, covS))
            snd_derivative += probs.snd_derivative(self.likelihood_type, (y, f, s))
            first_derivative += probs.fst_derivative(self.likelihood_type, (y, f, s))

        return first_derivative/self.k2, snd_derivative/self.k2

    def compute_expected_uis(self, othercols, otherdims):
        uis = np.ones((self.D,))
        for dim, col in enumerate(othercols):
            # Sample from the approximate posterior
            (mi, Si) = self.posterior.get_vector_distribution(otherdims[dim], col)
            uis = np.multiply(uis, mi)
        return uis

    def sample_uis(self, othercols, otherdims):
        uis = np.ones((self.D,))
        for dim, col in enumerate(othercols):
            # Sample from the approximate posterior
            (mi, Si) = self.posterior.get_vector_distribution(otherdims[dim], col)
            uj_sample = np.random.multivariate_normal(mi, Si)
            uis = np.multiply(uis, uj_sample)

        return uis

    def report_metrics(self, iteration, start, mean_change, cov_change):
        if iteration == self.report:
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
        # For each entry, find posterior distribution and
        # sample to compute the expected log likelihood
        # there is also the KL divergence term
        vlb = 0.0
        for i in range(len(values)):
            entry = entries[i]
            val   = values[i]
            m, S  = self.compute_posterior_param(entry)

    def estimate_expected_log_likelihood(self, m, S):
        return

    def compute_KL_divergence(self, m, S):
        return

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