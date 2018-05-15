import Probability.ProbFun as probs
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

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

        # optimization scheme
        self.opt_scheme = scheme

        # Stochastic optimization parameters
        self.batch_size  = 1 # batch size not needed for sparse data
        self.iterations  = 5000
        self.k1          = k1
        self.k2          = k2

        # adagrad parameters
        self.ada_acc_grad = [np.zeros((self.D, s)) for s in self.size_per_dim]
        self.eta = 1

        # schaul-like update window width
        self.window_size = 5
        self.recent_gradients_norm_sqr = [np.zeros((s, self.window_size)) for s in self.size_per_dim]
        self.recent_gradients_sum      = [np.zeros((s, self.window_size)) for s in self.size_per_dim]
        self.cur_gradient_pos          = [[0 for _ in range(s)] for s in self.size_per_dim]

    def factorize(self):
        """
        factorize
        :return: None
        Doing round robin updates for each column of the
        hidden matrices
        """
        update_column_pointer = [0] * self.order

        # while self.check_stop_cond():
        for iteration in range(self.iterations):
            if iteration != 0 and iteration % self.report == 0:
                print ("iteration: ", iteration, " - MSRE: ", self.evaluate())

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
        step_size  = self.find_step_size_mean_param(dim, i, m, meanGrad)
        m += np.multiply(step_size, meanGrad)

        self.model.q_posterior.update(dim, i, (m, S))

    def find_step_size_mean_param(self, dim, i, m, mGrad):
        if self.opt_scheme == "adagrad": # adaGrad
            self.ada_acc_grad[dim][:, i] += np.multiply(mGrad, mGrad)
            step_size = self.eta / np.sqrt(self.ada_acc_grad[dim][:, i])

        elif self.opt_scheme == "schaul": # Schaul-like update (before adaDelta)
            current_grad_pos  = self.cur_gradient_pos[dim][i]
            numGradients = min(current_grad_pos + 1, self.window_size)
            self.recent_gradients_norm_sqr[dim][i, current_grad_pos % self.window_size] = norm(mGrad) ** 2
            self.recent_gradients_sum[dim][i, current_grad_pos % self.window_size] = np.sum(mGrad)
            self.cur_gradient_pos[dim][i] += 1

            expected_gradient_squared = (np.sum(self.recent_gradients_sum[dim][i, :])/numGradients)**2
            expected_squares_gradient = np.sum(self.recent_gradients_norm_sqr[dim][i, :])/numGradients
            # step_size = self.eta * expected_gradient_squared / expected_squares_gradient
            # The above formula blows up T.T
            step_size   = self.eta / np.sqrt(expected_squares_gradient)

        return step_size

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
            f = probs.sample("normal", (meanf, covS))
            snd_derivative += probs.snd_derivative("normal", (y, f, s))
            first_derivative += probs.fst_derivative("normal", (y, f, s))
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

    def estimate_di(self, dim, i, coord, y):
        """
        :param i:
        :param dim:
        :param coord:
        :param y:

        :return:
        """
        other_j     = coord[: dim].append(coord[dim + 1 :])
        nlist       = list(range(self.order))
        other_dims  = (nlist[:dim]).append(nlist[dim + 1 : ])
        di          = np.zeros((self.D, 1))

        (m, S) = self.model.q_posterior.find(dim, i)
        invS   = np.linalg.inv(S)

        for _ in range(self.k1):
            ui = np.ones((self.D, 1))

            for d, j in enumerate(other_j):
                # Get the mean and posterior
                (mi,Si) = self.model.q_posterior.find(other_dims[d], j)
                sample = probs.sample("normal", (mi, Si))
                ui     = np.multiply(ui, sample)

            mf     = np.dot(ui, m)
            mS     = np.dot(ui, np.dot(invS, ui))
            acc    = 0.0
            p_params = self.model.p_likelihood

            for _ in range(self.k2):
                f  = probs.sample("normal", (mf, mS))
                acc += probs.fst_derivative("normal", (y, *p_params))

            di = di + ui * acc

        return di/self.k2

    def estimate_Di(self, dim, i, coord, y):
        """
        :param i: the id of the column
        :param dim: the id of the hidden factor
        :param coord: the coordinate of the observed_by_id entry
        :param y : the value of the tensor entry

        :return: D_i that is computed according to the formula in the paper
        """

        # Exclude the dimension to be updated
        other_j     = coord[: dim].append(coord[dim + 1 :])
        nlist       = list(range(self.order))
        other_dims  = (nlist[:dim]).append(nlist[dim + 1 : ])
        Di          = np.zeros((self.D, self.D))

        (m, S) = self.model.q_posterior.find(dim, i)
        invS   = np.linalg.inv(S)

        # For each coordinate
        for _ in range(self.k1):
            ui = np.ones((self.D, 1))
            for d, j in enumerate(other_j):
                # Get the mean and posterior
                (mi,Si) = self.model.q_posterior.find(other_dims[d], j)
                sample = probs.sample("normal", (mi, Si))
                ui     = np.multiply(ui, sample)

            # Get the currnet mean and covariance of the factor
            mf     = np.dot(ui, m)
            mS     = np.dot(ui, np.dot(invS, ui))
            acc    = 0.0
            p_params = self.model.p_likelihood

            for _ in range(self.k2):
                f  = probs.sample("normal", (mf, mS))
                acc += probs.snd_derivative("normal", (y, *p_params))

            Di = Di + np.outer(ui, ui) * acc # Update Di

        return Di/(2*self.k2)

    def check_stop_cond(self):
        """
        :return: boolean
        Check for stopping condition

        """
        return

    def evaluate(self):
        """
        :return:
        """
        error = 0.0
        for i, entry in enumerate(self.tensor.test_entries):
            predict = self.predict_entry(entry)
            correct = self.tensor.test_vals[i]
            error += np.abs(predict - correct)/abs(correct)
        return error/len(self.tensor.test_vals)

    def predict_likelihood(self, f):
        ptype = self.model.p_likelihood.type
        if ptype == "normal":
            return f
        elif ptype == "poisson":
            return np.log(1 + np.exp(f))
        elif ptype == "bernoulli":
            if f < 0.5:
                return 0
            return 1
        else:
            raise Exception("Unidentified likelihood type")

    def predict_entry(self, entry):
        u = np.ones((self.D,))
        for dim, col in enumerate(entry):
            m, S = self.model.q_posterior.find(dim, col)
            u = np.multiply(u, m)
        f = np.sum(u)
        return self.predict_likelihood(f)



