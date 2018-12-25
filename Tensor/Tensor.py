"""
tensor.py
Contains definitinos of different data structures that are
needed in data representation and storage.
"""
import  numpy as np
import time
from Probability import ProbFun as probs
from abc import abstractclassmethod, abstractmethod


class Tensor(object):
    def  __init__(self, datatype="real"):
        """
        :param datatype:
        :param binary_cutoff:
        """
        assert(datatype in ["real", "ordinal", "count", "binary"])
        self.datatype = datatype


    def read_from_file(self, filename, train_size, validation_size, test_size):
        """
        Read through the file
        into entries and then shuffle
        then choose train-test split
        """
        assert(validation_size + train_size + test_size == 1)

        f = open(filename, "r")
        headerread = False

        entries = []
        vals    = []
        for line in f:
            if not headerread:
                headerread = True
                dimensions = line.strip().split(",")
                self.dims  = [int(x) for x in dimensions]
                self.num_dimensions  = len(dimensions)
                continue

            data = line.strip().split(",")
            coord = [int(x) for x in data[:-1]]
            val   = float(data[-1])

            if self.datatype == "count":
                self.min_count = min(self.min_count, int(val))
                self.max_count = max(self.max_count, int(val))

            entries.append(coord)
            vals.append(val)

        # Generate random indices on this entries
        total_numentries = len(entries)
        train_num = int(total_numentries * train_size)
        valid_num = int(total_numentries * validation_size)
        test_num  = total_numentries - train_num - valid_num

        indices = np.arange(total_numentries)
        np.random.shuffle(indices)

        train_indices = indices[:train_num]
        valid_indices = indices[train_num : train_num + valid_num]
        test_indices  = indices[train_num + valid_num :]

        self.train_entries = np.take(entries, train_indices, axis=0)
        self.train_vals = np.take(vals, train_indices, axis=0)

        self.valid_entries = np.take(entries, valid_indices, axis=0)
        self.valid_vals = np.take(vals, valid_indices, axis=0)

        self.test_entries  = np.take(entries, test_indices, axis=0)
        self.test_vals = np.take(vals, test_indices, axis=0)


        self.observed_by_id = [[[] for _ in range(x)] for x in self.dims]

        for i in range(len(self.train_entries)):
            val = self.train_vals[i]
            coord = self.train_entries[i]
            #print("coord", coord)
            for dim, col in enumerate(coord):
                self.observed_by_id[dim][col].append((coord, val))


    """
    """
    def synthesize_data(self, dims, means, covariances, D=20, train=0.8, sparsity=1, noise=0.1, noise_ratio=True):
        """
        :param dims:
        :param means:
        :param covariances:
        :param D:
        :param train:
        :param sparsity:
        :return:
        """
        print("Generating synthetic", self.datatype , "valued data ... ")
        start = time.time()
        self.dims = dims
        self.D = D
        self.train = train
        self.noise = noise
        self.noise_ratio = noise_ratio

        self.matrices = self.generate_hidden_matrices(means, covariances)

        total         = np.prod(dims) # Total number of possible entries
        observed_num  = int(total * sparsity) # Number of observed_by_id entries
        train_size    = int(observed_num * train) # training set size

        self.observed_entries = self.generate_unique_coords(observed_num)
        self.observed_vals    \
            = self.organize_observed_entries_by_column(self.observed_entries, train_size)

        self.test_entries  = self.observed_entries[train_size :]
        self.test_vals     = self.observed_vals[train_size :]

        self.train_entries = self.observed_entries[: train_size]
        self.train_vals    = self.observed_vals[: train_size]

        end = time.time()
        print("Generating synthetic ", self.datatype, "valued data took: ", end- start)
        if self.datatype == "count":
            print("max_count = ", self.max_count, " min count = ", self.min_count)


    @abstractmethod
    def generate_hidden_matrices(self, means, covariances):
        raise NotImplementedError

    def reduce_train_size(self, train_ratio):
        """
        :param train_ratio:
        :return:
        """
        print("Using ", train_ratio, "of training data")
        new_train_size = int(self.train * train_ratio * len(self.observed_vals))

        self.train_entries = self.observed_entries[: new_train_size]
        self.train_vals    = self.observed_vals[: new_train_size]

        for dim, nrows in enumerate(self.dims):
            self.observed_by_id[dim] = [[] for _ in range(nrows)]

        for i in range(new_train_size):
            entry = self.observed_entries[i]
            f     = self.observed_vals[i]
            for dim, row_num in enumerate(entry):
                self.observed_by_id[dim][row_num].append((entry, f))

    def organize_observed_entries_by_column(self, observed_coords, train_size):
        ndim = len(self.dims)
        observed_vals = [0] * len(observed_coords)
        self.observed_by_id = [[] for _ in range(ndim)]

        for dim, nrows in enumerate(self.dims):
            self.observed_by_id[dim] = [[] for _ in range(nrows)]

        for entry_num, entry in enumerate(observed_coords):
            f = self.compute_entry_value(entry)
            observed_vals[entry_num] = f

            if entry_num < train_size:
                for dim in range(ndim):
                    row_num = entry[dim]
                    self.observed_by_id[dim][row_num].append((entry, f))

        return observed_vals

    def compute_entry_value(self, entry):
        ui = np.ones((self.D,))
        ndim = len(self.dims)
        for dim in range(ndim):
            row_num = entry[dim]
            ui = np.multiply(ui, self.matrices[dim][row_num, :])

        m = np.sum(ui)
        # print(m)
        if self.noise != 0:
            if self.noise_ratio: # If noise level is relative to m
                stddev = self.noise * np.abs(m)
            else: # If noise level is fixed stddev
                stddev = self.noise
            f = np.random.normal(m, stddev)
        else:
            f = m

        return self.data_link_fun(f)


    @abstractmethod
    def data_link_fun(self, f):
        raise NotImplementedError


    def organize_observed_entries(self, observed_num, train_size, dims, matrices, noise):
        ndim = len(dims)
        observed_entries = self.generate_unique_coords(observed_num)

        observed_vals          = [0] * len(observed_entries)
        self.observed_by_id    = [[] for _ in range(ndim)]

        for dim in range(ndim):
            nrows = dims[dim]
            self.observed_by_id[dim] = [[] for _ in range(nrows)]

        if noise != 0:
            s_array = np.random.normal(0, noise, size=(len(observed_entries,)))

        for entry_num, entry in enumerate(observed_entries):
            ui = np.ones_like(matrices[0][0, :])
            for dim in range(ndim):
                row_num  = entry[dim]
                ui  = np.multiply(ui, matrices[dim][row_num, :])

            m = np.sum(ui)
            if noise != 0:
                s = s_array[entry_num]
            else:
                s = 0

            f = self.actual_value(m + s)

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

        elif self.datatype == "count":
            f = self.link_fun(m)
            x = int(np.rint(f))
            self.max_count = max(self.max_count, x)
            self.min_count = min(self.min_count, x)
            return x
        else:
            raise NotImplementedError


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

    def load_data(self, filename):
        """
        :param filename:
        :return:
        """
        # TODO: implement when testing on real life datasets
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

    def evaluate_true_params(self):
        print("Evaluation for true params: ")
        print(" test_rsme | train_rsme | rel-te-err | rel-tr-err | train_nll | test_nll  |")

    def evaluate_error(self, entries, vals):
        rsme = 0.0
        error = 0.0

        for i in range(len(entries)):
            predict = self.predict_entry(entries[i])
            correct = vals[i]
            rsme += np.square(predict - correct)

            if self.likelihood_type == "normal":
                error += np.abs(predict - correct)/abs(correct)
            elif self.likelihood_type == "bernoulli":
                error += 1 if predict != correct else 0
            elif self.likelihood_type == "poisson":
                error += np.abs(predict - correct)

        rsme = np.sqrt(rsme/len(vals))
        error = error/len(vals)
        return rsme, error

    def save_tensor_data(self, data_folder):
        save_file = os.path.join(data_folder, "tensor.pickled")

