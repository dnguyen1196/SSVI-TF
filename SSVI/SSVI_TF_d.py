import Probability.ProbFun as probs
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
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
        self.report     = 1

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
            self.link_fun = lambda m : np.log(1 + np.exp(-m))

        # optimization scheme
        self.opt_scheme = scheme

        self.time_step = [1 for _ in range(self.order)]

        # Stochastic optimization parameters
        self.batch_size  = batch_size
        self.iterations  = 6000
        self.k1          = k1
        self.k2          = k2

        # adagrad parameters
        self.offset = 0.0001
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.size_per_dim]
        self.eta = 1

    def factorize(self):
        """
        factorize
        :return: None
        Doing round robin updates for each column of the
        hidden matrices
        """
        update_column_pointer = [0] * self.order
        start = time.time()
        # while self.check_stop_cond():
        for iteration in range(self.iterations):
            current = time.time()
            if iteration != 0 and iteration % self.report == 0:
                print ("iteration: ", iteration, " - test error: ", \
                       self.evaluate_test_error(), " - train error: ", self.evaluate_train_error(), " - time: ", current - start)

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
            observed_i   = np.take(observed_i, observed_idx, axis=0)

        M = len(observed_i)
        (m, S) = self.model.q_posterior.find(dim, i)

        Di_acc = np.zeros((self.D, self.D))
        di_acc = np.zeros((self.D, ))

        for entry in observed_i:
            coord  = entry[0]
            y      = entry[1]

            if self.likelihood_type == "normal":
                (di_acc_update, Di_acc_update) = self.estimate_di_Di_normal(dim, i, coord, y, m, S)
            else:
                (di_acc_update, Di_acc_update) = self.estimate_di_Di(dim, i, coord, y, m, S)

            Di_acc += Di_acc_update
            di_acc += di_acc_update

        Di_acc *= len(observed_i)/ min(self.batch_size, len(observed_i))
        di_acc *= len(observed_i)/ min(self.batch_size, len(observed_i))

        # Update covariance parameter
        rhoS = self.rho_cov(iteration)
        covGrad = (1./self.pSigma[dim] * np.eye(self.D) - 2 * Di_acc)
        S = inv((1-rhoS) * inv(S) + rhoS * covGrad)

        # Update mean parameter
        meanGrad = (np.inner(1./self.pSigma[dim] * np.eye(self.D), self.pmu - m) + di_acc)
        if self.opt_scheme == "sgd":
            m_update = self.rho(iteration) * meanGrad
        else:
            m_update = self.compute_update_mean_param(dim, i, m, meanGrad)

        m += m_update
        # print(np.linalg.norm(m_update))
        self.model.q_posterior.update(dim, i, (m, S))

    def compute_update_mean_param(self, dim, i, m, mGrad):
        """

        :param dim: dimension of the hidden matrix
        :param i: column of hidden matrix
        :param m: current value of mean parameter
        :param mGrad: computed gradient
        :return:

        Compute the update for the mean parameter dependnig on the
        optimization scheme
        """
        self.ada_acc_grad[dim][:, i] += np.multiply(mGrad, mGrad)
        return self.eta / np.sqrt(self.ada_acc_grad[dim][:, i]) * mGrad

    def estimate_di_Di_normal(self, dim, i, coord, y, mui, Sui):
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

            Expected_fst_derivative, Expected_snd_derivative = \
                self.compute_expected_first_snd_derivative(y, meanf, covS)

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

    def compute_expected_first_snd_derivative(self, y, meanf, covS) :
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
            uj_sample = probs.sample("multivariate_normal", (mi, Si))
            uis = np.multiply(uis, uj_sample)
        return uis

    def check_stop_cond(self):
        """
        :return: boolean
        Check for stopping condition
        """
        return

    def evaluate_train_error(self):
        """
        :return:
        """
        error = 0.0
        for i, entry in enumerate(self.tensor.train_entries):
            predict = self.predict_entry(entry)
            correct = self.tensor.train_vals[i]
            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            else:
                return 0

        return error/len(self.tensor.train_vals)

    def evaluate_test_error(self):
        """
        :return:
        """
        error = 0.0
        for i in range(len(self.tensor.test_entries)):
            predict = self.predict_entry(self.tensor.test_entries[i])
            correct = self.tensor.test_vals[i]

            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                # if predict != correct:
                #     print ("errror: ", predict, correct)
                error += 1 if predict != correct else 0
            else:
                return 0
        return error/len(self.tensor.test_vals)

    def predict_y_given_m(self, m):
        if self.likelihood_type == "normal":
            return m
        elif self.likelihood_type == "poisson":
            f = self.link_fun(m)
            # TODO: implement Gauss-Hermite quadrature


            return 1
        elif self.likelihood_type == "bernoulli":
            return 1 if m >= 0 else -1
        else:
            raise Exception("Unidentified likelihood type")

    def compute_expected_count(self, hermite_weights):


        return

    def compute_gauss_hermite(self, f, n):

        return

    def predict_entry(self, entry):
        u = np.ones((self.D,))
        for dim, col in enumerate(entry):
            m, S = self.model.q_posterior.find(dim, col)
            u = np.multiply(u, m)
        m = np.sum(u)
        return self.predict_y_given_m(m)



