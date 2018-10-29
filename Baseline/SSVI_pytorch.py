import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
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


class SSVI_torch(torch.nn.Module):
    def __init__(self, tensor, rank=10, lambd=0.01):
        super().__init__()

        self.tensor = tensor
        self.dims = tensor.dims
        self.rank = rank
        self.lambd = lambd
        self.datatype = tensor.datatype

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
        self.round_robins_indices = [0 for _ in self.dims]
        self.k1 = 32

    def factorize(self, maxiters, lr=1):
        # optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.Adagrad(self.parameters(), lr=lr)
        # optimizer = optim.RMSprop(self.parameters(), lr=lr)
        if self.datatype == "real":
            optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif self.datatype == "binary":
            optimizer = optim.Adagrad(self.parameters(), lr=1)
        elif self.datatype == "count":
            optimizer = optim.Adagrad(self.parameters(), lr=1)

        with torch.enable_grad():
            for iteration in range(maxiters):
                for dim, col in enumerate(self.round_robins_indices):
                    optimizer.zero_grad()

                    observed = self.tensor.find_observed_ui(dim, col)

                    if len(observed) > self.batch_size:
                        observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
                        observed_subset = np.take(observed, observed_idx, axis=0)
                    else:
                        observed_subset = observed

                    loss = self.loss_fun(observed_subset, dim, col)
                    loss.backward()
                    col = Variable(torch.LongTensor(col))
                    optimizer.step()

                if iteration in [0, 5, 10, 50] or iteration % 100 == 0:
                    self.evaluate(iteration)

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] += 1
                    self.round_robins_indices[dim] %= self.dims[dim]

    def loss_fun(self, observed, dim, col):
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
        batch_expectation = self.compute_batch_expectation_term(entries, ys, dim, col)

        # Compute the KL term as a batch
        batch_kl = self.compute_batch_kl_term(entries)

        loss -= batch_expectation + self.lambd * batch_kl

        return loss

    def compute_batch_expectation_term(self, entries, ys, vdim, vcol):
        num_samples = len(entries)
        ndim = len(self.dims)

        element_mult_samples = torch.ones(num_samples, self.k1, self.rank) # shape = (num_samples, k1, rank)
        for dim, ncol in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            all_cols = Variable(torch.LongTensor(all_cols))
            if dim == vdim:
                all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
                all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)
            else:
                all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
                # all_ms.detach()
                all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)
                # all_Ls.detach()

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
        if self.datatype == "binary":
            expected_log_pdf[expected_log_pdf == -float("Inf")] = -1000
            # expected_log_pdf[expected_log_pdf == float("Inf")]  = 1000

        elif self.datatype == "count":
            expected_log_pdf[expected_log_pdf == -float("Inf")] = -100
            expected_log_pdf[torch.isnan(expected_log_pdf)] = -100
            expected_log_pdf[expected_log_pdf == float("Inf")]  = 100

        batch_expectation = expected_log_pdf.sum()
        return batch_expectation

    def compute_entry_expectation_term(self, entry, y):
        # E_{q(u)q(v)} [log p(y|f)]
        fs_samples = torch.ones(self.k1, self.rank)
        for dim, col in enumerate(entry):
            colid = Variable(torch.LongTensor([col]))
            m = self.means[dim](colid)[0]
            L = self.chols[dim](colid)[0]

            # Generate n_samples of eps
            epsilons = self.standard_multi_normal.sample((self.k1,)) # (num_sample, rank)

            # Apply reparameterization
            # ui = mui + eps * (L**2)
            for i in range(self.k1):
                fs_samples[i, :] *= m + epsilons[i, :] * L **2

        # fij samples
        fs = fs_samples.sum(1)

        # TODO: implement log pdf for different data type
        # Here, we just have log pdf for the normal case
        log_pdf = -0.5 * (fs - y)**2
        return log_pdf.mean(0)

    def compute_entry_kl_term(self, entry):
        # TODO: KL term computation can be done outside
        # For now, to ensure correctness
        kl = 0.
        for dim, col in enumerate(entry):
            col = Variable(torch.LongTensor([col]))
            m   = self.means[dim](col)[0]
            L   = self.chols[dim](col)[0]
            S   = L**2
            kl -= 0.5 * torch.sum(1 + torch.log(S**2) - m ** 2 - S**2)
        return kl

    def compute_batch_kl_term(self, entries):
        kl = 0.
        for dim, ncol in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = self.means[dim](all_cols) # shape = (num_samples, rank)
            all_Ls = self.chols[dim](all_cols) # shape = (num_samples, rank)
            all_S  = all_Ls**2

            kl -= 0.5 * torch.sum(1 + torch.log(all_S ** 2) - all_ms**2 - all_S**2)
        return kl

    def compute_log_pdf(self, fs_samples, target_matrix):
        # fs_samples.shape = (num_samples, k1)
        # target_matrix.shape = (num_samples, k1)
        if self.datatype == "real":
            exp_term = fs_samples - target_matrix  # (num_samples, k1) - (num_samples,k1)
            exp_term_squared = exp_term ** 2
            log_pdf = -0.5 * exp_term_squared  # shape = (num_samples, k1)

        elif self.datatype == "binary":
            yf_product = fs_samples * target_matrix
            log_pdf = F.logsigmoid(yf_product)
            # sigmoid = torch.sigmoid(yf_product)
            # log_pdf = torch.log(sigmoid)

        elif self.datatype == "count":
            # yf_product = fs_samples * target_matrix
            # exp_f      = torch.exp(fs_samples)
            # # log_pdf = 1/ torch.factorial(target_matrix) * torch.exp(-exp_f) * torch.exp(yf_product)
            # log_pdf = torch.exp(-exp_f) * torch.exp(yf_product)
            neg_log_exp_f = -fs_samples
            log_fs = torch.log(fs_samples)
            log_fs[torch.isnan(log_fs)] = -1000
            log_pow_f_y = target_matrix * torch.log(fs_samples)
            log_pdf = neg_log_exp_f + log_pow_f_y
            log_pdf /= 1e6

        return log_pdf

    def evaluate(self, iteration):
        if iteration == 0:
            print(" iteration | test rsme | train rsme |")

        train_rsme, _ = self.evaluate_train_error()
        test_rsme, _ = self.evaluate_test_error()
        print ("{:^10} {:^10} {:^10}".format(iteration, test_rsme, train_rsme))

    def evaluate_train_error(self):
        rsme, error = self.evaluate_rsme(self.tensor.train_entries, self.tensor.train_vals)
        return rsme, error

    def evaluate_test_error(self):
        rsme, error = self.evaluate_rsme(self.tensor.test_entries, self.tensor.test_vals)
        return rsme, error

    def evaluate_rsme(self, entries, vals):
        """
        :param entries:
        :param vals:
        :return:
        """
        rsme = 0.0
        error = 0.0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            predict = self.predict_entry(entry)
            correct = vals[i]
            rsme += (predict - correct)**2

        rsme = np.sqrt(rsme/num_entries)
        error = error/num_entries
        return rsme, error

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