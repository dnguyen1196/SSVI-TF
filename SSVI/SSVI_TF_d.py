import Probability.ProbFun as probs
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import math
import time

"""
SSVI_TF.py
SSVI algorithm to learn the hidden matrix factors behind tensor
factorization
"""
class H_SSVI_TF_2d():
    def __init__(self, model, tensor, rank, rho_cov, k1=1, k2=10, scheme="adagrad", batch_size=5):
        """
        :param model: the generative model behind factorization
        :param tensor:
        :param rank:
        :param rho_cov: step size formula for covariance parameter
        :param k1: number of samples for hidden column vectors
        :param k2: number of samples for f variables
        :param scheme: optimization scheme
        """
        self.model      = model
        self.tensor     = tensor
        self.D          = rank

        self.size_per_dim = tensor.dims        # dimension of the tensors
        self.order        = len(tensor.dims)   # number of dimensions
        self.rho_cov      = rho_cov

        # Get prior mean and covariance
        self.pmu = np.ones((self.D,))
        self.pSigma = [1 for _ in self.size_per_dim]
        self.likelihood_type = model.p_likelihood.type

        if self.likelihood_type == "normal":
            self.link_fun = lambda m : m
        elif self.likelihood_type == "bernoulli":
            self.link_fun = lambda m : probs.sigmoid(m)
        elif self.likelihood_type == "poisson":
            self.max_count   = tensor.max_count
            self.min_count   = tensor.min_count
            self.herm_degree = 50
            self.hermite_points, self.hermite_weights = np.polynomial.hermite.hermgauss(self.herm_degree)
            self.link_fun = lambda m : probs.poisson_link(m)

        # optimization scheme
        self.opt_scheme = scheme
        self.time_step = [1 for _ in range(self.order)]

        # Stochastic optimization parameters
        self.batch_size  = batch_size
        self.iterations  = 60001
        self.k1          = k1
        self.k2          = k2
        self.epsilon     = 0.0001

        # adagrad parameters
        self.offset = 0.000001
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.size_per_dim]
        self.ada_acc_cov  = [np.zeros((self.D, self.D, s)) for s in self.size_per_dim]
        self.eta = 1
        self.poisson_eta = 0.05

        # keep track of changes in norm
        self.norm_changes = [np.ones((s, 2)) for s in self.size_per_dim]

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
            if max(mean_change, cov_change) < self.epsilon:
                break

            current = time.time()
            if iteration != 0 and iteration % self.report == 0:
                print ("iteration: ", iteration, " - test error: ", \
                       self.evaluate_test_error(), " - train error: ",\
                       self.evaluate_train_error(), " - max mean change: ",\
                       mean_change, " - max cov change: ", cov_change,\
                       " - time: ", current - start)

            for dim in range(self.order):
                col = update_column_pointer[dim]
                # Update the natural params of the col-th factor
                # in the dim-th dimension
                self.update_natural_params(dim, col, iteration)
                self.update_hyper_parameter(dim, iteration)

            # Move on to the next column of the hidden matrices
            for dim in range(self.order):
                if (update_column_pointer[dim] + 1 == self.size_per_dim[dim]):
                    self.time_step[dim] += 1  # increase time step
                update_column_pointer[dim] = (update_column_pointer[dim] + 1) \
                                             % self.size_per_dim[dim]

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
        (m, S) = self.model.q_posterior.find(dim, i)

        Di_acc = np.zeros((self.D, self.D))
        di_acc = np.zeros((self.D,))

        for entry in observed_i:
            coord = entry[0]
            y = entry[1]

            if self.likelihood_type == "normal":
                (di_acc_update, Di_acc_update) = self.estimate_di_Di_complete_conditional(dim, i, coord, y, m, S)
            else:
                (di_acc_update, Di_acc_update) = self.estimate_di_Di(dim, i, coord, y, m, S)

            Di_acc += Di_acc_update
            di_acc += di_acc_update

        Di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))
        di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))

        # Update covariance parameter
        covGrad = (1. / self.pSigma[dim] * np.eye(self.D) - 2 * Di_acc)
        covStep = self.compute_stepsize_cov_param(dim, i, covGrad)

        # TODO: updating the cholesky decomposition, not the actual covariance matrix
        # TODO: but now the step size is constant anyway
        S_next = inv((np.ones_like(covGrad) - covStep) * inv(S) + np.multiply(covStep, covGrad))

        # Update mean parameter, NOTE that this using pSigma is the identity
        meanGrad = (np.inner(1. / self.pSigma[dim] * np.eye(self.D), self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
        m_next   = m + np.multiply(meanStep, meanGrad)

        mean_change = np.linalg.norm(m_next - m)
        cov_change  = np.linalg.norm(S_next - S, 'fro')
        self.norm_changes[dim][i, :] = np.array([mean_change, cov_change])
        self.model.q_posterior.update(dim, i, (m_next, S_next))

    def compute_stepsize_cov_param(self, dim, i, covGrad):
        if self.likelihood_type != "poisson":
            return 0.01

        return self.poisson_eta

        acc_grad = self.ada_acc_cov[dim][:, :, i]
        grad_squared = np.square(covGrad)
        self.ada_acc_cov[dim][:, :, i] += grad_squared
        return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_squared)))

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
        if self.likelihood_type != "poisson":
            return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))
        else:
            return np.divide(self.poisson_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

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
        s = self.model.p_likelihood.params

        for j, d in enumerate(otherdims):
            m, S = self.model.q_posterior.find(d, othercols[j])
            d_acc = np.multiply(d_acc, m)
            D_acc = np.multiply(D_acc, S + np.outer(m, m))

        Di = -1./s * D_acc
        di = y/s * d_acc - 1./s * np.inner(D_acc, mui)
        return di, Di

    def estimate_di_Di(self, dim, i, coord, y, m, S):
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

        for k1 in range(self.k1):
            ui = self.sample_uis(othercols, otherdims)

            meanf     = np.dot(ui, m)
            covS     = np.dot(ui, np.inner(S, ui))

            # print(meanf)

            Expected_fst_derivative, Expected_snd_derivative = \
                self.estimate_expected_derivative(y, meanf, covS)

            di += ui * Expected_fst_derivative/self.k1                   # Update di
            Di += np.outer(ui, ui) * Expected_snd_derivative/(2*self.k1) # Update Di

        return di, Di

    def update_hyper_parameter(self, dim, iteration):
        """
        :param dim:
        :return:
        """
        sigma = 0.0
        M = self.size_per_dim[dim]
        for j in range(M):
            m, S = self.model.q_posterior.find(dim, j)
            sigma += np.trace(S) + np.dot(m, m)
        self.pSigma[dim] = sigma/(M*self.D)

    def estimate_expected_derivative(self, y, meanf, covS) :
        first_derivative = 0.0
        snd_derivative = 0.0
        s = self.model.p_likelihood.params

        for k2 in range(self.k2):
            f = probs.sample("normal", (meanf, covS))
            snd_derivative += probs.snd_derivative(self.likelihood_type, (y, f, s))
            first_derivative += probs.fst_derivative(self.likelihood_type, (y, f, s))

        return first_derivative/self.k2, snd_derivative/self.k2

    def compute_expected_uis(self, othercols, otherdims):
        uis = np.ones((self.D,))
        for dim, col in enumerate(othercols):
            # Sample from the approximate posterior
            (mi, Si) = self.model.q_posterior.find(otherdims[dim], col)
            uis = np.multiply(uis, mi)
        return uis

    def sample_uis(self, othercols, otherdims):
        uis = np.ones((self.D,))
        for dim, col in enumerate(othercols):
            # Sample from the approximate posterior
            (mi, Si) = self.model.q_posterior.find(otherdims[dim], col)
            # try:
            uj_sample = probs.sample("multivariate_normal", (mi, Si))
            uis = np.multiply(uis, uj_sample)
            # except Warning:
            #     print(min(np.linalg.eigvals(Si)))
        return uis

    def check_stop_cond(self):
        """
        :return: boolean
        Check for stopping condition
        """
        max_mean_change = 0
        max_cov_change  = 0
        for dim in range(len(self.size_per_dim)):
            for col in range(self.size_per_dim[dim]):
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
            mi, Ci = self.model.q_posterior.find(dim, i)
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
                m, _ = self.model.q_posterior.find(dim, col)
                u = np.multiply(u, m)
            m = np.sum(u)
            return self.predict_y_given_m(m)
        else:
            # if predicting count values, the calculations are more involved
            # as we first need find the parameters of the posterior distribution
            m, S = self.compute_posterior_param(entry)
            # print(probs.poisson_link(m), S)
            return np.rint(m)
            # res  = self.compute_expected_count(m, S)
            # return np.rint(res)