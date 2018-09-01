from Tensor.Tensor import Tensor
from Probability import ProbFun as probs
import numpy as np

class count_tensor(Tensor):
    def __init__(self):
        super(count_tensor, self).__init__(datatype="count")

        self.link_fun = lambda m: probs.poisson_link(m)
        self.min_count = 100
        self.max_count = 0

    def generate_hidden_matrices(self, mean=None, cov=None):
        ndim = len(self.dims)
        mean_array = np.linspace(0, 3, ndim)
        # Generate the random hidden matrices
        matrices = [[] for _ in range(ndim)]

        for i in range(ndim):
            mean = np.ones((self.D,)) * mean_array[i]
            cov  = np.eye(self.D)     * 0.1
            matrices[i] = self.create_random_matrix(self.dims[i], self.D, mean, cov)

        return matrices

    def data_link_fun(self, m):
        f = self.link_fun(m)
        #x = np.rint(f) # NOTE: change data generation
        x = np.random.poisson(f)
        self.max_count = max(self.max_count, x)
        self.min_count = min(self.min_count, x)
        return x

    def actual_value(self, m):
        return self.link_fun(m)
