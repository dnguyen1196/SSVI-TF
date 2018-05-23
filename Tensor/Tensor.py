"""
tensor.py
Contains definitinos of different data structures that are
needed in data representation and storage.
"""
import  numpy as np
import time

class tensor(object):
    def  __init__(self, datatype="real", binary_cutoff=0.0):
        """

        :param datatype:
        :param binary_cutoff:
        """
        assert(datatype in ["real", "ordinal", "count", "binary"])
        self.datatype = datatype
        if self.datatype == "real":
            self.link_fun = lambda m : m
        elif self.datatype == "binary":
            self.link_fun = lambda m : 1. / (1. + np.exp(-m))
            self.binary_cutoff = binary_cutoff
        elif self.datatype == "count":
            self.link_fun = lambda m : np.log(1 + np.exp(-m))

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
        print("Generating synthetic data ... ")
        start = time.time()
        self.dims = dims

        ndim = len(dims)
        matrices = [[]] * ndim
        # Generate the random hidden matrices
        for i in range(ndim):
            matrices[i] = self.create_matrix(dims[i], D, means[i], covariances[i])

        total         = np.prod(dims) # Total number of possible entries
        observed_num  = int(total * sparsity) # Number of observed_by_id entries
        train_size    = int(observed_num * train) # training set size

        observed_entries, observed_vals \
            = self.organize_observed_entries(observed_num, train_size, dims, matrices)

        self.test_entries  = observed_entries[train_size :]
        self.test_vals     = observed_vals[train_size :]
        self.train_entries = observed_entries[: train_size]
        self.train_vals    = observed_vals[: train_size]

        if self.datatype == "binary":
            num_ones = np.sum(self.test_vals)
            num_minus = (len(self.test_vals) -  num_ones)/2
            print("There are: ", num_minus, " -1's value out of ", len(self.test_vals), " test values")

        # self.verify_binary_entries(matrices)

        end = time.time()
        print("Generating synthetic data took: ", end- start)

    def organize_observed_entries(self, observed_num, train_size, dims, matrices):
        ndim = len(dims)
        observed_entries = self.generate_unique_coords(observed_num)

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

            m = np.sum(ui)
            f = self.actual_value(m)
            if self.datatype == "binary":
                assert(f * (m- self.binary_cutoff) > 0)

            observed_vals[entry_num] = f
            if entry_num < train_size:
                for dim in range(ndim):
                    row_num = entry[dim]
                    self.observed_by_id[dim][row_num].append((entry, f))

        return observed_entries, observed_vals

    def actual_value(self, m):
        if self.datatype == "real":
            return m
        elif self.datatype == "binary":
            return 1 if m >= self.binary_cutoff else -1
        else:
            return None

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

    def generate_unique_coords(self, num):
        total = np.prod(self.dims)
        idx = np.random.choice(total, num, replace=False)

        divs = []
        for i, s in enumerate(self.dims):
            otherdims = self.dims[i+1 : ]
            divs.append(np.prod(otherdims))

        unique_coords = []
        for id in idx:
            unique_coords.append(self.id_to_coordinate(id, divs))
        return unique_coords

    def id_to_coordinate(self, id, divs):
        coord = []
        id = float(id)
        for dim, size in enumerate(divs):
            k = int(id/size)
            coord.append(k)
            id -= size * k
        return coord

    # TODO: implement
    def load_data(self, filename):
        """
        :param filename: 
        :return: 
        """
        pass

    def verify_binary_entries(self, matrices):
        for i in range(len(self.train_vals)):
            actual = self.train_vals[i]
            entry  = self.train_entries[i]
            ui     = np.ones_like(matrices[0][0, :])
            ndim   = len(entry)
            for dim in range(ndim):
                row_num  = entry[dim]
                ui  = np.multiply(ui, matrices[dim][row_num, :])
            m = np.sum(ui)
            f = self.actual_value(m)
            assert (f == actual)