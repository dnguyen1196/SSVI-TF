import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
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


class CPDecompPytorch3D(torch.nn.Module):
    def __init__(self, tensor, rank=10, lambd=0.01):
        """
        :param tensor:
        :param init:
        :param lambd:
        :param rank:

        """
        super().__init__()

        self.tensor = tensor
        self.dims   = tensor.dims
        self.rank = rank
        self.lambd = lambd
        self.datatype = tensor.datatype

        factors = []
        for dim, ncol in enumerate(self.dims):
            factors.append(torch.nn.Embedding(ncol, rank))

        self.factors = ListModule(*factors)
        self.batch_size = 128
        self.round_robins_indices = [0 for _ in self.dims]

    def factorize(self, num_iters):
        optimizer = optim.Adagrad(self.parameters(), lr=1)

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
                    loss.backward()
                    optimizer.step()

                if iteration in [0, 5, 10, 50] or iteration % 100 == 0:
                    self.evaluate(iteration)

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] = \
                        (1 + self.round_robins_indices[dim]) % self.dims[dim]

    def loss_fun(self, observed):
        """
        :param observed:
        :return:
        """
        loss = 0.
        entries = [pair[0] for pair in observed]
        y = Variable(torch.FloatTensor([pair[1] for pair in observed]))

        inners = torch.ones(len(observed), self.rank)

        for dim, _ in enumerate(self.dims):
            all_cols = [x[dim] for x in entries]
            all_cols = Variable(torch.LongTensor(all_cols))
            vectors = self.factors[dim](all_cols) # A list of vector
            inners *= vectors
            loss += self.lambd * torch.sum(vectors**2) # Regularization term

        loss += torch.sum((torch.sum(inners, dim=1) - y)**2)

        return loss

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

    def predict_entry(self, entry):
        inner = torch.ones(self.rank)
        for dim, col in enumerate(entry):
            col = Variable(torch.LongTensor([np.long(col)]))
            inner *= self.factors[dim](col)[0]

        if self.datatype == "real":
            return float(torch.sum(inner))
        elif self.datatype == "binary":
            return 1 if torch.sum(inner) > 0 else -1
        elif self.datatype == "count":
            return float(torch.sum(inner))

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
