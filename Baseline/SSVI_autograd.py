import autograd.numpy as np
from autograd.numpy.random import multivariate_normal
from builtins import range
from autograd import grad


def adam(grad, x, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adapted from autograd.misc.optimizers"""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x

def rmsprop(grad, x, callback=None, num_iters=100, step_size=0.1, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x

def adagrad(grad, x, callback=None, num_iters=100, step_size=0.1, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x)) #* 0.0001
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad + g**2
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)

    return x



class SSVI_autograd():
    def __init__(self, tensor, gradient_update="S", rank=10):
        """
        """
        self.tensor = tensor
        self.num_train = len(tensor.train_vals)
        self.dims = tensor.dims
        self.ndim = len(self.dims)
        self.rank = rank
        self.datatype = tensor.datatype
        self.gradient_update = gradient_update

        # Hyperparameters
        self.sigma = 1
        self.batch_size = 64
        self.lambd = 1/self.batch_size # Constant for the KL terms
        self.round_robins_indices = [0 for _ in self.dims]
        self.k1 = 32

        # Approximate parameters
        self.means = [[] for _ in self.ndims]
        self.chols = [[] for _ in self.ndims]

        for dim, nrow in enumerate(self.ndims):
            for _ in range(nrow):
                self.means[dim].append(np.ones((self.rank,)))
                self.chols[dim].append(np.ones((self.rank,)))

        self.params = {"means" : self.means, "chols:" : self.chols}

    def factorize(self, maxiters, report=[], interval=50, round_robins=True):
        """

        :param maxiters:
        :param algorithm:
        :param lr:
        :param report:
        :param interval:
        :return:
        """

        start = 0
        expectation_term = 0.
        kl_term = 0.


        def callback(phi, iteration, gradient):
            """
            Call back function
            """
            if iteration in report or iteration % interval == 0:
                self.evaluate(iteration, 0, 0) # To replace with expectation and KL term

        gradient = grad(self.compute_batch_ELBO)
        self.params = adagrad(gradient, x, callback=callback, num_iters=100, step_size=0.1, eps=10**-8)


    def compute_batch_ELBO(self, params, iteration):
        """

        """
        for dim, col in enumerate(self.round_robins_indices):
            # optimizer.zero_grad()

            # Get mini-batch in round-robbins
            if round_robins:
                observed_subset, num_observed_i = self.get_mini_batch_round_robins(dim, col)
            # Get mini-batch from random order
            else:
                observed_subset = self.get_mini_batch(start)
                start = (start + self.batch_size) % len(self.tensor.train_vals)

            expectation_term, kl_term = self.elbo_compute(observed_subset)
            num_sample = len(observed_subset)


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
            all_ms = torch.stack([self.means[dim][row] for row in all_rows], dim=0)
            all_Ls = torch.stack([self.chols[dim][row] for row in all_rows], dim=0)

            sampler = MultivariateNormal(torch.zeros(self.rank), torch.eye(self.rank))
            epsilon_tensor = sampler.sample((num_samples, self.k1))
            # epsilon_tensor = self.standard_multi_normal.sample((num_samples, self.k1)) # shape = (num_sample, k1, rank)

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

        for dim, _ in enumerate(self.dims):
            all_rows = [x[dim] for x in entries]
            # all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = torch.stack([self.means[dim][row] for row in all_rows], dim=0)
            all_Ls = torch.stack([self.chols[dim][row] for row in all_rows], dim=0)
            all_S = all_Ls ** 2

            kl_batch = 0.5 * torch.sum(-self.rank + torch.sum(all_S, dim=1) + torch.sum(all_ms**2, dim=1) - torch.log(1/torch.prod(all_S, dim=1)))

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
        # log_pdf = torch.max(log_pdf, torch.FloatTensor([-100]))
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


    def evaluate(self, iteration, expectation, kl):
        if iteration == 0:
            print(" iteration | test mae  | train mae |  E-term  |    KL   |")

        train_mae = self.evaluate_train_error()
        test_mae = self.evaluate_test_error()
        # expectation_term = expectation.detach().numpy()
        # kl_term = kl.detach().numpy()
        print ("{:^10} {:^10} {:^10} {:^10} {:^10}".format(iteration, np.around(test_mae, 4), np.around(train_mae, 4), \
                                                    expectation, kl))

    def evaluate_train_error(self):
        if self.datatype == "binary":
            return self.evaluate_error_rate(self.tensor.train_entries, self.tensor.train_vals)
        return self.evaluate_mae(self.tensor.train_entries, self.tensor.train_vals)

    def evaluate_test_error(self):
        if self.datatype == "binary":
            return self.evaluate_error_rate(self.tensor.test_entries, self.tensor.test_vals)
        return self.evaluate_mae(self.tensor.test_entries, self.tensor.test_vals)

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

    def evaluate_error_rate(self, entries, vals):
        """
        :param entries:
        :param vals:
        :return:
        """
        error = 0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            predict = self.predict_entry(entry)
            correct = vals[i]
            if correct != predict:
                error += 1

        return float(error)/num_entries

    def predict_entry(self, entry):
        # TODO: generalize to other likelihood types
        inner = np.ones((self.rank,))
        for dim, col in enumerate(entry):
            # inner *= self.means[dim][col, :]
            inner *= self.means[dim][col]

        if self.datatype == "real":
            return float(np.sum(inner))
        elif self.datatype == "binary":
            return 1 if np.sum(inner) > 0 else -1
        elif self.datatype == "count":
            return float(np.sum(inner))

    def get_mini_batch(self, start):
        end = (start + self.batch_size) % len(self.tensor.train_vals)
        if end > start:
            observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, end)]
        else:
            observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, len(self.tensor.train_vals))]
            observed_subset.extend([(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(end)])
        return observed_subset

    def get_mini_batch_round_robins(self, dim, col):
        observed = self.tensor.find_observed_ui(dim, col)
        if len(observed) > self.batch_size:
            observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
            observed_subset = np.take(observed, observed_idx, axis=0)
        else:
            observed_subset = observed
        return observed_subset, len(observed)



