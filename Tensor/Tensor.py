"""
tensor.py
Contains definitinos of different data structures that are
needed in data representation and storage.
"""
import  numpy as np

class tensor(object):
    def __init__(self):
        return

    # synthesize data according to some model
    def synthesize(self, dims, means, covariances, D=10, train=0.8, sparsity=0.5):
        """
        :param dims:
        :param means:
        :param covariances:
        :param D:
        :param train:
        :param sparsity:
        :return:
        """
        self.dims = dims

        ndim = len(dims)
        self.matrices = [[]] * ndim
        # Generate the random hidden matrices
        for i in range(ndim):
            self.matrices[i] = self.create_matrix(dims[i], D, means[i], covariances[i])

        total         = np.prod(dims) # Total number of possible entries
        observed_num  = int(total * sparsity) # Number of observed_by_id entries
        train_size    = int(observed_num * train) # training set size

        observed_entries, observed_vals \
            = self.organize_observed_entries(observed_num, train_size, dims, self.matrices)

        self.test_entries  = observed_entries[train_size :]
        self.test_vals     = observed_vals[train_size :]
        self.train_entries = observed_entries[: train_size]
        self.train_vals    = observed_vals[: train_size]

    def organize_observed_entries(self, observed_num, train_size, dims, matrices):
        ndim = len(dims)
        observed_entries = []
        unique = set()

        while len(observed_entries) < observed_num:
            # For each dimension, pick a hidden vector randomly
            rand_indices = [np.random.choice(dims[n]) for n in range(ndim)]
            uniq_id = "".join(str(e) for e in rand_indices)
            if not (uniq_id in unique):
                observed_entries.append(rand_indices)
                unique.add(uniq_id)

        observed_vals          = [0] * len(observed_entries)
        self.observed_by_id    = [[] for _ in range(ndim)]

        for dim in range(ndim):
            nrows = dims[dim]
            self.observed_by_id[dim] = [[] for _ in range(nrows)]

        for entry_num, entry in enumerate(observed_entries):
            ui = np.ones_like(matrices[0][0, :])
            for dim in range(ndim):
                row_num  = entry[dim]
                ui  = np.multiply(ui, matrices[dim][row_num, :])

            y = np.sum(ui)
            observed_vals[entry_num] = y
            if entry_num < train_size:
                for dim in range(ndim):
                    row_num = entry[dim]
                    self.observed_by_id[dim][row_num].append((entry, y))

        return observed_entries, observed_vals

    def create_matrix(self, nrow, ncol, m, S):
        """
        :param nrow:
        :param ncol:
        :param m:
        :param S:
        :return:
        """
        matrix = np.zeros((nrow, ncol))
        for i in range(nrow):
            # Populate each row with a random vector generated from mean m,
            # and covariance S
            matrix[i, :] = np.random.multivariate_normal(m, S)
        return matrix

    def find_observed_ui(self, dim, i):
        """
        :param dim: dimension number of the hidden matrix (U1, U2, ...)
        :param i:   column number
        :return: the list of observed_by_id entries involving this column
        """
        hidden_matrix = self.observed_by_id[dim]
        return hidden_matrix[i]


    # TODO: implement
    def load_data(self, filename):
        """
        :param filename: 
        :return: 
        """
        pass