import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import ModuleList, ParameterList, Parameter
import copy

from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.distributions.kl as KL

import numpy as np
import math


"""

Natural parameter update

Stores
(diagonal covariance)
S, m

Still need to compute
dT/dS and dT/dm

mean parameter update:
inv(S)m = inv(S)m + p (m - inv(S)m - dT/dS m + dT/dS)
>>> multiplying both sides with S
m  =  m + p(Sm - m - S dT/dS m + S dT/dS)

Covariance parameter update:
0.5 inv(S) = 0.5 inv(S) + p (S - 0.5 inv (S) + dT/dS)

>>> Update formula
S = inv(S + p (2S - inv(S) + 2dT/dS)

The question is 
>> how to run backward() and then modify the computed gradient value
>> Loop through the means and covs value
>> 

"""

class SSVI_torch(torch.nn.Module):
    def __init__(self, tensor, using_natural_gradient="S", rank=10):
        super().__init__()

        self.tensor = tensor
        self.num_train = len(tensor.train_vals)
        self.dims = tensor.dims
        self.ndim = len(self.dims)
        self.rank = rank
        self.datatype = tensor.datatype
        self.using_natural_gradient = using_natural_gradient

        self.means = ModuleList()
        self.chols = ModuleList()

        for dim, ncol in enumerate(self.dims):
            mean_list = ParameterList()
            cov_list  = ParameterList()
            for _ in range(ncol):
                mean_list.append(Parameter(torch.randn(rank), requires_grad=True))
                cov_list.append(Parameter(torch.randn(rank), requires_grad=True))

            self.means.append(mean_list)
            self.chols.append(cov_list)

        self.standard_multi_normal = MultivariateNormal(torch.zeros(rank), torch.eye(rank))
        self.sigma = 1
        self.batch_size = 128
        self.lambd = 1/self.batch_size
        self.round_robins_indices = [0 for _ in self.dims]
        self.k1 = 32

    def factorize(self, maxiters, algorithm="AdaGrad", lr=1, report=[], interval=50):
        if algorithm == "AdaGrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr)
        elif algorithm == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr)

        with torch.enable_grad():
            start = 0

            for iteration in range(maxiters):
                if iteration in report or iteration % interval == 0:
                    self.evaluate(iteration)

                for dim, col in enumerate(self.round_robins_indices):
                    optimizer.zero_grad()

                    # Get mini-batch from random order
                    # end = (start + self.batch_size) % len(self.tensor.train_vals)
                    # if end > start:
                    #     observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, end)]
                    # else:
                    #     observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, len(self.tensor.train_vals))]
                    #     observed_subset.extend([(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(end)])
                    # start = end

                    # Get mini-batch in round-robbins
                    observed = self.tensor.find_observed_ui(dim, col)
                    if len(observed) > self.batch_size:
                        observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
                        observed_subset = np.take(observed, observed_idx, axis=0)
                    else:
                        observed_subset = observed

                    expectation_term, kl_term = self.loss_function_compute(observed_subset)
                    num_sample = len(observed_subset)
                    num_observed_i = len(observed)

                    # Natural gradient
                    if self.using_natural_gradient == "N":
                        self.natural_gradient_update(observed_subset, expectation_term, kl_term, optimizer)
                        # self.natural_gradient_update_round_robin(expectation_term, kl_loss, optimizer, num_sample, num_observed_i, dim, col)

                    # Hybrid update
                    elif self.using_natural_gradient == "H":
                        self.hybrid_gradient_update(observed_subset, expectation_term, kl_term, optimizer)

                    # Standard gradient update
                    else:
                        loss = -self.num_train/self.batch_size * expectation_term + kl_term
                        loss.backward()
                        optimizer.step()
                        # self.standard_gradient_update_round_robin(expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col)
                        # self.standard_gradient_sanity_check(observed_subset, expectation_term, kl_term, optimizer)

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] += 1
                    self.round_robins_indices[dim] %= self.dims[dim]


    def standard_gradient_update_round_robin(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        loss = -self.num_train/self.batch_size * expectation_term + kl_term
        loss.backward()
        dm = copy.deepcopy(self.means._parameters[str(dim)].grad[col, :])
        dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[col, :])

        # TODO: note that if I set dm, dL the way above, it does gets updated with the zero_grad!!!
        optimizer.zero_grad() # Remove the grad of other factors
        self.means._parameters[str(dim)].grad[col, :] = dm
        self.chols._parameters[str(dim)].grad[col, :] = dL
        optimizer.step()

    def natural_gradient_update_round_robin(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        """

        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:
        """

        # Automatically compute the derivative with respect to the parameters
        expectation_term.backward()

        dm = copy.deepcopy(self.means._parameters[str(dim)].grad[col, :])#
        dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[col, :])
        m = copy.deepcopy(self.means._parameters[str(dim)].data[col, :])
        L = copy.deepcopy(self.chols._parameters[str(dim)].data[col, :])

        # Compute the natural gradient required for natural parameter
        # Note that pytorch does MINUS gradient * stepsize
        # and we're doing gradient ascent
        # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1

        # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
        scale = num_observed_i / num_sample
        G_mean = dm - dL * m / L
        natural_mean_grad = (m - m / L ** 2 + scale * G_mean) * -1

        G_chol = -dL * 1 / (2 * L)
        natural_chol_grad = (L ** 2 - 0.5 * 1 / L ** 2 + scale * G_chol) * -1

        # Precompute
        # m -> m/L**2
        self.means._parameters[str(dim)].data[col, :] = m / L ** 2
        # L -> 0.5 /L**2
        self.chols._parameters[str(dim)].data[col, :] = 0.5 * L ** 2

        optimizer.zero_grad() # Remove the grad of other factors
        # Replace the gradient with the natural gradient
        self.means._parameters[str(dim)].grad[col, :] = natural_mean_grad
        self.chols._parameters[str(dim)].grad[col, :] = natural_chol_grad
        optimizer.step()

        # The current covariance parameter being stored is 0.5/L**2
        # 0.5/L**2 = x => L = sqrt(0.5/x)
        L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[col, :])
        L_squared = 0.5 / L_natural
        L_squared = F.relu(L_squared) + 1e-4
        # L_squared = torch.max(L_squared, torch.FloatTensor([1e-4]))

        # The current mean parameter being stored is m/L**2
        # m/L**2 =   => m = L**2 x
        m_natural = copy.deepcopy(self.means._parameters[str(dim)].data[col, :])
        m_new = L_squared * m_natural
        L_new = torch.sqrt(L_squared)
        # L[torch.isnan(L)] = 0.1
        self.means._parameters[str(dim)].data[col, :] = m_new
        self.chols._parameters[str(dim)].data[col, :] = L_new

    def natural_gradient_update(self, observed_subset, expectation_term, kl_term, optimizer):
        """

        :param observed_subset:
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:

        Mean update
        1/(L**2)m  = 1/(L**2)m + p [ m - 1/(L**2) + dT/dm + dT/dL m/L ]
        0.5 / L**2 = 0.5 / L**2 + p [ L**2 - 1/(2L**2) - dT/dL 1/(2L) ]


        """
        entries = [pair[0] for pair in observed_subset]
        # Compute the gradient of the expectation term with respect to the
        # parameters of the model
        # Let pytorch compute the necessary gradient of the expectation term
        # with respect to the parameters
        # loss = -expectation_term
        expectation_term.backward()
        # loss.backward()
        # loss = -expectation_term + kl_term
        # loss.backward()

        L_prev = list()

        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)

            # Get the GRADIENT of the expectation term wrt
            # the MEAN and CHOLESKY parameters
            dm = copy.deepcopy(self.means._parameters[str(dim)].grad[all_cols, :])
            dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_cols, :])
            m  = copy.deepcopy(self.means._parameters[str(dim)].data[all_cols, :])
            L  = copy.deepcopy(self.chols._parameters[str(dim)].data[all_cols, :])

            L_prev.append(L)

            # Compute the natural gradient required for natural parameter
            # Note that pytorch does MINUS gradient * stepsize
            # and we're doing gradient ascent
            # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1

            # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
            scale = 1
            G_mean =  dm - dL * m / L
            natural_mean_grad = (m - m / L ** 2 + scale * G_mean) * -1

            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/L**2 + scale * G_chol) * -1

            # Precompute the current natural parameters
            # m -> m/L**2
            self.means._parameters[str(dim)].data[all_cols, :] = m/L**2
            # L -> 0.5 /L**2
            self.chols._parameters[str(dim)].data[all_cols, :] = 0.5 * L**2

            # Replace the gradient with the natural gradient
            self.means._parameters[str(dim)].grad[all_cols, :] = natural_mean_grad
            self.chols._parameters[str(dim)].grad[all_cols, :] = natural_chol_grad

        # Do one step of update
        optimizer.step()

        # Re-update the parameters
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)

            # The current covariance parameter being stored is 0.5/L**2
            # 0.5/L**2 = x => L = sqrt(0.5/x)
            L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[all_cols, :])
            L_squared = 0.5/L_natural
            # L_squared = F.relu(L_squared) + 1e-4
            L_squared = torch.max(L_squared, torch.FloatTensor([1e-4]))

            # The current mean parameter being stored is m/L**2
            # m/L**2 =   => m = L**2 x
            m_natural = copy.deepcopy(self.means._parameters[str(dim)].data[all_cols, :])
            m_new = L_squared * m_natural

            L_new = torch.sqrt(L_squared)
            # L[torch.isnan(L)] = 0.1
            self.means._parameters[str(dim)].data[all_cols, :] = m_new
            self.chols._parameters[str(dim)].data[all_cols, :] = L_new


    def hybrid_gradient_update(self, observed_subset, expectation_term, kl_term, optimizer):
        # If using natural gradient
        entries = [pair[0] for pair in observed_subset]

        # Batch loss
        batch_loss = - self.num_train/self.batch_size * expectation_term + kl_term
        batch_loss.backward(retain_graph=True)
        mean_grad = list()

        # Zeros out the Cholesky factor
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)
            # Get the GRADIENT of total loss with respect to the CHOLESKY factors
            dm = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_cols, :])
            mean_grad.append(dm)

        # Update the cholesky factor via natural gradient
        # Loss term from expectation
        optimizer.zero_grad()

        expectation_term.backward()
        # Update the Cholesky factors via Natural gradient
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)

            # Get the GRADIENT of the expectation term wrt
            # CHOLESKY parameters
            dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_cols, :])
            L  = copy.deepcopy(self.chols._parameters[str(dim)].data[all_cols, :])

            # Replace the gradient for the mean with previously computed value
            self.means._parameters[str(dim)].grad[all_cols, :] = mean_grad[dim]

            # Compute the natural gradient required for natural parameter
            scale = 1
            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/L**2 + scale * G_chol) * -1

            # Replace the gradient with the natural gradient
            self.chols._parameters[str(dim)].grad[all_cols, :] = natural_chol_grad
            # Compute the current natural parameters
            self.chols._parameters[str(dim)].data[all_cols, :] = 1/L**2

        # Do one step of update
        optimizer.step()

        # Re-Flip the Cholesky decomposition
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)
            L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[all_cols, :])
            L_squared = 0.5/L_natural
            L_new     = torch.sqrt(F.relu(L_squared)) + 1e-4
            self.chols._parameters[str(dim)].data[all_cols, :] = L_new

    def standard_gradient_sanity_check(self, observed_subset, expectation_term, kl_term, optimizer):
        """

        :param observed_subset:
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :return:
        """
        # If using natural gradient
        entries = [pair[0] for pair in observed_subset]

        # Batch loss
        batch_loss = - self.num_train/self.batch_size * expectation_term + kl_term
        batch_loss.backward(retain_graph=True)
        mean_grad = list()

        # Stores the gradient of mean parameters
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)
            # Get the GRADIENT of total loss with respect to the CHOLESKY factors
            dm = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_cols, :])
            mean_grad.append(dm)

        optimizer.zero_grad()

        batch_loss.backward()
        # Compute the gradient of the cholesky factors and put back the mean gradient
        for dim, ncol in enumerate(self.dims):
            all_cols = set([x[dim] for x in entries])
            all_cols = list(all_cols)
            self.means._parameters[str(dim)].grad[all_cols, :] = mean_grad[dim]

        # TODO: theoretically, this should produce the same results as standard gradient update
        # Do one step of update
        optimizer.step()

    def loss_function_compute(self, observed):
        """
        :param observed:
        :return:
        """
        entries = [pair[0] for pair in observed]
        ys = Variable(torch.FloatTensor([pair[1] for pair in observed]))

        # entries = list of coordinates
        # y = vector of entry values
        # Compute the expectation as a batch
        batch_expectation = self.compute_batch_expectation_term(entries, ys)

        # Compute the KL term as a batch
        batch_kl = self.compute_batch_kl_term(entries)

        # loss -= self.num_train/self.batch_size * batch_expectation + (1/ (self.batch_size * self.ndim)) * batch_kl
        expectation_term = batch_expectation
        kl_loss          =  (1 / (self.batch_size )) * batch_kl

        return expectation_term, kl_loss

    def compute_batch_expectation_term(self, entries, ys):
        num_samples = len(entries)
        ndim = len(self.dims)

        element_mult_samples = torch.ones(num_samples, self.k1, self.rank) # shape = (num_samples, k1, rank)
        for dim, nrow in enumerate(self.dims):
            all_rows = [x[dim] for x in entries]
            # all_cols = Variable(torch.LongTensor(all_cols))

            # all_ms = self.means[dim][all_cols, :] # shape = (num_samples, rank)
            # all_Ls = self.chols[dim][all_cols, :] # shape = (num_samples, rank)

            # TODO: stack these rows into a matrix
            all_ms = [self.means[dim][row] for row in all_rows]
            all_Ls = [self.chols[dim][row] for row in all_rows]

            epsilon_tensor = self.standard_multi_normal.sample((num_samples, self.k1)) # shape = (num_sample, k1, rank)

            # How to create k1 copies (rows of all_ms)
            all_ms.unsqueeze_(-1)
            ms_copied = all_ms.expand(num_samples, self.rank, self.k1)
            # ms_copied = all_ms.repeat(1, self.k1)  # shape = (num_samples, k1, rank)
            ms_copied = ms_copied.transpose(2, 1)
            for num in range(num_samples):
                L_squared = all_Ls[num, :]**2 # shape = (rank)
                eps_term  =  epsilon_tensor[num, :, :] # shape = (k1, rank)
                var_term  = eps_term * L_squared # shape = (k1, rank)
                element_mult_samples[num, :, :] *= ms_copied[num, :, :] + var_term

        # fs_samples.shape = (num_samples, k1)
        fs_samples = element_mult_samples.sum(dim=2) # sum along the 3rd dimension (along rank)

        target_vector = Variable(torch.FloatTensor(ys))
        target_vector = target_vector.view(num_samples, 1)
        target_matrix = target_vector.repeat(1, self.k1)

        # Compute log pdf
        log_pdf = self.compute_log_pdf(fs_samples, target_matrix)

        expected_log_pdf = log_pdf.mean(dim=1)

        batch_expectation = expected_log_pdf.sum()
        return batch_expectation

    def compute_batch_kl_term(self, entries):
        kl = 0.
        for dim, ncol in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            # all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = self.means[dim][all_cols, :] # shape = (num_samples, rank)
            all_Ls = self.chols[dim][all_cols, :] # shape = (num_samples, rank)
            all_S  = all_Ls ** 2

            kl_div = KL._kl_normal_normal(Normal(all_ms, all_S), Normal(0, 1))
            kl_div = torch.sum(kl_div)

            kl -= kl_div
            # kl -= 0.5 * torch.sum(1 + torch.log(all_S ** 2) - all_ms**2 - all_S**2)
        return kl

    def compute_log_pdf(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        # fs_samples.shape = (num_samples, k1)
        # target_matrix.shape = (num_samples, k1)
        if self.datatype == "real":
            return self.compute_log_pdf_normal(fs_samples, target_matrix)

        elif self.datatype == "binary":
            return self.compute_log_pdf_bernoulli(fs_samples, target_matrix)

        elif self.datatype == "count":
            return self.compute_log_pdf_poisson(fs_samples, target_matrix)

        return log_pdf

    def compute_log_pdf_normal(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        dist = Normal(fs_samples, 1)
        log_pdf = dist.log_prob(target_matrix)
        return log_pdf

    def compute_log_pdf_bernoulli(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        dist = Bernoulli(torch.sigmoid(fs_samples))
        log_pdf = dist.log_prob(target_matrix)
        return log_pdf

    def compute_log_pdf_poisson(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:

        This gives the most trouble because of
        """
        lamb = F.relu(fs_samples, inplace=True)
        # Compute log likelihood
        dist = Poisson(lamb)
        log_pdf = dist.log_prob(target_matrix)

        nan_indices = torch.isnan(log_pdf)
        log_pdf[nan_indices] = -100
        inf_indices = torch.isinf(log_pdf)
        log_pdf[inf_indices] = -100
        neg_inf_indices = torch.isinf(-log_pdf)
        log_pdf[neg_inf_indices] = -100

        return log_pdf

    def evaluate(self, iteration):
        if iteration == 0:
            print(" iteration | test mae  | train mae |")

        train_mae = self.evaluate_train_error()
        test_mae = self.evaluate_test_error()
        print ("{:^10} {:^10} {:^10}".format(iteration, test_mae, train_mae))

    def evaluate_train_error(self):
        mae = self.evaluate_mae(self.tensor.train_entries, self.tensor.train_vals)
        return mae

    def evaluate_test_error(self):
        mae = self.evaluate_mae(self.tensor.test_entries, self.tensor.test_vals)
        return mae

    def evaluate_mae(self, entries, vals):
        """
        :param entries:
        :param vals:
        :return:
        """
        mae = 0.0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            predict = self.predict_entry(entry)
            correct = vals[i]
            mae += abs(predict - correct)

        mae = mae/num_entries
        return mae

    def predict_entry(self, entry):
        # TODO: generalize to other likelihood types
        inner = torch.ones(self.rank)
        for dim, col in enumerate(entry):
            # inner *= self.means[dim][col, :]
            inner *= self.means[dim][col]

        if self.datatype == "real":
            return float(torch.sum(inner))
        elif self.datatype == "binary":
            return 1 if torch.sum(inner) > 0 else -1
        elif self.datatype == "count":
            return float(torch.sum(F.relu(inner)))