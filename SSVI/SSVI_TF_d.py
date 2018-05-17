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
    def __init__(self, model, tensor, rank, rho_cov, k1=1, k2=10, scheme="adagrad"):
        """
        :param model:
        :param tensor:
        :param rank:
        """
        self.model      = model
        self.tensor     = tensor
        self.D          = rank
        self.report     = 1

        self.size_per_dim = tensor.dims        # dimension of the tensors
        self.order        = len(tensor.dims)   # number of dimensions
        self.rho_cov      = rho_cov

        # Get prior mean and covariance
        # self.pmu, self.pSigma = self.model.p_prior.find(0, 0)
        # self.pmu = [np.ones((self.D, )) for _ in len(self.size_per_dim)]
        self.pmu = np.ones((self.D,))
        self.pSigma = [1 for _ in self.size_per_dim]

        self.likelihood_type = model.p_likelihood.type

        if self.likelihood_type == "normal":
            self.link_fun = lambda m : m
        elif self.likelihood_type == "bernoulli":
            self.link_fun = lambda m : 1. /(1 + np.exp(-m))
        elif self.likelihood_type == "poisson":
            self.link_fun = lambda m : np.log(1 + np.exp(-m))

        # optimization scheme
        self.opt_scheme = scheme

        # Stochastic optimization parameters
        self.batch_size  = 1 # batch size not needed for sparse data
        self.iterations  = 5000
        self.k1          = k1
        self.k2          = k2
        if scheme == "adagrad":
            # adagrad parameters
            self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.size_per_dim]
            self.eta = 1

        elif scheme == "schaul":
            # schaul-like update window width
            self.window_size = 5
            self.recent_gradients\
                = [[np.zeros((self.D, self.window_size)) for _ in range(s)] for s in self.size_per_dim]

            self.recent_gradients_sum \
                = [[np.zeros((self.D, self.window_size)) for _ in range(s)] for s in self.size_per_dim]
            self.cur_gradient_pos = [[0 for _ in range(s)] for s in self.size_per_dim]
        elif scheme == "adadelta":
            # adadelta parameters
            self.gamma = 0.1
            self.offset = 0.0001
            self.alpha = 1
            self.g_t = [[np.zeros((self.D,)) for _ in range(s)] for s in self.size_per_dim]
            self.s_t = [[np.zeros((self.D,)) for _ in range(s)] for s in self.size_per_dim]

            # nesterov accelerated gradient
            self.momentum = 0.9
            self.delta_theta_t = [[np.zeros((self.D,)) for _ in range(s)] for s in self.size_per_dim]

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
                self.update_natural_params(dim, col)
                self.update_hyper_parameter(dim)

            # Move on to the next column of the hidden matrices
            for dim in range(self.order):
                update_column_pointer[dim] = (update_column_pointer[dim] + 1) \
                                             % self.size_per_dim[dim]

    def update_natural_params(self, dim, i):
        """
        :param i:
        :param dim:
        :return:
        """
        observed_i = self.tensor.find_observed_ui(dim, i)
        # list of observed_by_id entries

        M = len(observed_i)
        (m, S) = self.model.q_posterior.find(dim, i)

        Di_acc = np.zeros((self.D, self.D))
        di_acc = np.zeros((self.D, ))

        for entry in observed_i:
            coord  = entry[0]
            y      = entry[1]
            (di_acc_update, Di_acc_update) = self.estimate_di_Di(dim, i, coord, y, m, S)
            Di_acc += Di_acc_update
            di_acc += di_acc_update

        # Update covariance parameter
        rhoS = self.rho_cov
        covGrad = (1./self.pSigma[dim] * np.eye(self.D) - 2 * Di_acc)
        S = inv((1-rhoS) * inv(S) + rhoS * covGrad)

        # Update mean parameter
        meanGrad = (np.inner(1./self.pSigma[dim] * np.eye(self.D), self.pmu - m) + di_acc)
        update   = self.compute_update_mean_param(dim, i, m, meanGrad)
        m = np.add(update, m)

        self.model.q_posterior.update(dim, i, (m, S))

    def compute_update_mean_param(self, dim, i, m, mGrad):
        if self.opt_scheme == "adagrad":
            self.ada_acc_grad[dim][:, i] += np.multiply(mGrad, mGrad)
            return self.eta / np.sqrt(self.ada_acc_grad[dim][:, i]) * mGrad

        elif self.opt_scheme == "schaul":
            current_grad_pos = self.cur_gradient_pos[dim][i]
            self.recent_gradients[dim][i][: , current_grad_pos] = mGrad
            self.cur_gradient_pos[dim][i] = (self.cur_gradient_pos[dim][i] + 1) % self.window_size

            recent_gradients         = self.recent_gradients[dim][i]
            recent_gradients_squared = np.square(recent_gradients)
            recent_gradients_sum     = np.sum(recent_gradients, 1)

            expected_squares_gradient = np.sum(recent_gradients_squared, 1)
            step_size = np.multiply(np.square(recent_gradients_sum))

            return self.eta / np.sqrt(expected_squares_gradient) * mGrad

        elif self.opt_scheme == "adadelta":
            g_0 = self.g_t[dim][i]
            s_0 = self.s_t[dim][i]

            # Update g_t
            g_t = (1. - self.gamma) * np.square(mGrad) + self.gamma * g_0
            self.g_t[dim][i] = g_t

            # Compute gradient update
            delta_theta_t = self.alpha * \
                        np.divide(np.sqrt(np.add(s_0, self.offset)), \
                                  np.sqrt(np.add(g_t, self.offset))) * mGrad

            # Update s_t
            self.s_t[dim][i] = (1 - self.gamma) * np.square(delta_theta_t) + self.gamma * s_0

            # Update deltaTheta_t
            self.delta_theta_t[dim][i] = delta_theta_t
            return delta_theta_t

    def estimate_di_Di(self, dim, i, coord, y, m, S):
        """
        :param dim:
        :param i:
        :param coord:
        :param y:
        :return:
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

    def update_hyper_parameter(self, dim):
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

    def compute_expected_first_snd_derivative(self, y, meanf, covS):
        first_derivative = 0.0
        snd_derivative = 0.0
        s = self.model.p_likelihood.params

        for k2 in range(self.k2):
            f = probs.sample(self.likelihood_type, (meanf, covS))
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
        for i, entry in enumerate(self.tensor.test_entries):
            predict = self.predict_entry(entry)
            correct = self.tensor.test_vals[i]
            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            else:
                return 0

        return error/len(self.tensor.test_vals)

    def predict_y_given_m(self, m):
        if self.likelihood_type == "normal":
            return m

        elif self.likelihood_type == "poisson":
            f = self.link_fun(m)
            #TODO: implement
            return 1

        elif self.likelihood_type == "bernoulli":
            # print(m)
            # f = self.link_fun(m)
            return 1 if m >= 0.5 else -1
        else:
            raise Exception("Unidentified likelihood type")

    def predict_entry(self, entry):
        u = np.ones((self.D,))
        for dim, col in enumerate(entry):
            m, S = self.model.q_posterior.find(dim, col)
            u = np.multiply(u, m)
        m = np.sum(u)
        return self.predict_y_given_m(m)



