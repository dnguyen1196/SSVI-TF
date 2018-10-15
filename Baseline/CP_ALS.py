import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class CPDecompPytorch3D(torch.nn.Module):
    def __init__(self, tensor, lambd=1, rank=10):
        """

        :param tensor:
        :param init:
        :param lambd:
        :param rank:

        Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        """
        super().__init__()

        self.tensor = tensor
        self.dims   = tensor.dims
        self.rank = rank

        self.d0_factors = torch.nn.Embedding(self.dims[0], rank, sparse=True)
        self.d1_factors = torch.nn.Embedding(self.dims[1], rank, sparse=True)
        self.d2_factors = torch.nn.Embedding(self.dims[2], rank, sparse=True)

        self.factors = [self.d0_factors, self.d1_factors, self.d2_factors]
        self.batch_size = 128
        self.round_robins_indices = [0 for _ in self.dims]

    def factorize(self, num_iters):
        optimizer = optim.SGD(self.parameters(), lr=0.00000001)
        with torch.enable_grad():
            for iteration in range(num_iters):
                for dim, col in enumerate(self.round_robins_indices):
                    optimizer.zero_grad()

                    observed = self.tensor.find_observed_ui(dim, col)

                    if len(observed) > self.batch_size:
                        observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
                        observed_subset = np.take(observed, observed_idx, axis=0)
                    else:
                        observed_subset = observed

                    loss = self.loss_fun(observed_subset)
                    # print(loss)
                    loss.backward()
                    optimizer.step()

                if iteration in [0, 5, 10, 50] or iteration % 100 == 0:
                    self.evaluate(iteration)

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] += 1
                    self.round_robins_indices[dim] %= self.dims[dim]


    def loss_fun(self, observed):
        """

        :param observed:
        :return:

        TODO: other types of tensor entries
        """
        # with torch.enable_grad():
        loss = 0.
        for entry, y in observed:
            inner = torch.ones(self.rank)
            for dim, col in enumerate(entry):
                col = Variable(torch.LongTensor([np.long(col)]))
                vector = self.factors[dim](col)[0]

                inner *= vector
            # print(loss)
            loss += (torch.sum(inner) - y)**2

        return loss

    def evaluate(self, iteration):
        if iteration == 0:
            print(" iteration | test rsme | train rsme |")

        train_rsme, _ = self.evaluate_train_error()
        test_rsme, _ = self.evaluate_test_error()
        # print ("{:^10} {:^10}".format(np.around(test_rsme, 4), np.around(train_rsme, 4)))
        print ("{:^10} {:^10} {:^10}".format(iteration, test_rsme, train_rsme))


    def evaluate_train_error(self):
        rsme, error = self.evaluate_RSME(self.tensor.train_entries, self.tensor.train_vals)
        return rsme, error

    def evaluate_test_error(self):
        rsme, error = self.evaluate_RSME(self.tensor.test_entries, self.tensor.test_vals)
        return rsme, error

    def evaluate_RSME(self, entries, vals):
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

            # if self.likelihood_type == "normal":
            #     error += np.abs(predict - correct)/abs(correct)
            # elif self.likelihood_type == "bernoulli":
            #     error += 1 if predict != correct else 0
            # elif self.likelihood_type == "poisson":
            #     error += np.abs(predict - correct)

        rsme = torch.sqrt(rsme/num_entries)
        error = error/num_entries
        return rsme, error

    def predict_entry(self, entry):
        inner = torch.ones(self.rank)
        for dim, col in enumerate(entry):
            col = Variable(torch.LongTensor([np.long(col)]))
            inner *= self.factors[dim](col)[0]
        return torch.sum(inner)
