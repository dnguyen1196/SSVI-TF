import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.distributions.kl as KL

import numpy as np
import math


class ListParams(torch.nn.Module):
    def __init__(self, *args):
        super(ListParams, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


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
    def __init__(self, tensor, using_natural_gradient=False, rank=10):
        super().__init__()

        self.tensor = tensor
        self.num_train = len(tensor.train_vals)
        self.dims = tensor.dims
        self.ndim = len(self.dims)
        self.rank = rank
        self.datatype = tensor.datatype
        self.using_natural_gradient = using_natural_gradient

        means = []
        covs  = []
        for dim, ncol in enumerate(self.dims):
            means.append(torch.nn.Embedding(ncol, rank))
            covs.append(torch.nn.Embedding(ncol, rank))

        self.means = ListParams(*means)
        self.chols = ListParams(*covs)
        self.standard_multi_normal = MultivariateNormal(torch.zeros(rank), torch.eye(rank))

        self.sigma = 1
        self.batch_size = 128
        self.lambd = 1/self.batch_size
        self.round_robins_indices = [0 for _ in self.dims]
        self.k1 = 32

    def factorize(self, maxiters, lr=1):
        # optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.Adagrad(self.parameters(), lr=lr)
        # optimizer = optim.RMSprop(self.parameters(), lr=lr)
        if self.datatype == "real":
            optimizer = optim.Adagrad(self.parameters(), lr=lr)
            # optimizer = optim.SGD(self.parameters(), lr=0.01)

        elif self.datatype == "binary":
            optimizer = optim.Adagrad(self.parameters(), lr=1)
            # optimizer = optim.SGD(self.parameters(), lr=0.1)
        elif self.datatype == "count":
            optimizer = optim.Adagrad(self.parameters(), lr=1)
            # optimizer = optim.SGD(self.parameters(), lr=0.0000000000001)

        with torch.enable_grad():
            start = 0

            for iteration in range(maxiters):
                if iteration in [0, 1, 5, 10, 50] or iteration % 100 == 0:
                    self.evaluate(iteration)

                for dim, col in enumerate(self.round_robins_indices):
                    optimizer.zero_grad()

                    # Get mini-batch
                    end = (start + self.batch_size) % len(self.tensor.train_vals)
                    if end > start:
                        observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, end)]
                    else:
                        observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, len(self.tensor.train_vals))]
                        observed_subset.extend([(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(end)])
                    start = end

                    loss = self.loss_fun(observed_subset)
                    loss.backward()

                    if self.using_natural_gradient:
                        self.natural_gradient_update(observed_subset)

                    optimizer.step()

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] += 1
                    self.round_robins_indices[dim] %= self.dims[dim]

    def natural_gradient_update(self, observed_subset):
        # If using natural gradient
        entries = [pair[0] for pair in observed_subset]

        for dim, ncol in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
            all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)

    def loss_fun(self, observed):
        """
        :param observed:
        :return:
        """
        loss = torch.zeros(1)
        entries = [pair[0] for pair in observed]
        ys = Variable(torch.FloatTensor([pair[1] for pair in observed]))

        # entries = list of coordinates
        # y = vector of entry values
        # Compute the expectation as a batch
        batch_expectation = self.compute_batch_expectation_term(entries, ys)

        # Compute the KL term as a batch
        batch_kl = self.compute_batch_kl_term(entries)

        # loss -= self.num_train/self.batch_size * batch_expectation + (1/ (self.batch_size * self.ndim)) * batch_kl
        loss -= self.num_train / self.batch_size * batch_expectation + (1 / (self.batch_size )) * batch_kl

        return loss

    def compute_batch_expectation_term(self, entries, ys):
        num_samples = len(entries)
        ndim = len(self.dims)

        element_mult_samples = torch.ones(num_samples, self.k1, self.rank) # shape = (num_samples, k1, rank)
        for dim, ncol in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
            all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)

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
            all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
            all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)
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
        # exp_f = torch.exp(fs_samples)
        # lamb  = torch.log(1 + exp_f)
        # # lamb  = torch.log(fs_samples)
        #
        # nan_indices = torch.isnan(lamb)
        # lamb[nan_indices] = fs_samples[nan_indices]
        # neg_indices = fs_samples < 0
        # lamb = fs_samples
        # lamb[neg_indices] = 0.
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
            col = Variable(torch.LongTensor([col]))
            inner *= self.means[dim](col)[0]

        if self.datatype == "real":
            return float(torch.sum(inner))
        elif self.datatype == "binary":
            return 1 if torch.sum(inner) > 0 else -1
        elif self.datatype == "count":
            return float(torch.sum(inner))