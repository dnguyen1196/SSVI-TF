from Tensor.Tensor import Tensor
from Probability import ProbFun as probs
import numpy as np

class binary_tensor(Tensor):
    def __init__(self, binary_cutoff=0.):
        super(binary_tensor, self).__init__(datatype="binary")
        self.binary_cutoff = binary_cutoff

    def generate_hidden_matrices(self, mean=None, cov=None):
        ndim = len(self.dims)
        mean_array = np.linspace(-1, 1, ndim)
        # Generate the random hidden matrices
        matrices = [[] for _ in range(ndim)]

        for i in range(ndim):
            mean = np.ones((self.D,)) * mean_array[i]
            cov  = np.eye(self.D)     * 0.1
            matrices[i] = self.create_random_matrix(self.dims[i], self.D, mean, cov)

        return matrices

    def data_link_fun(self, m):
        return 1 if m >= self.binary_cutoff else -1

    def actual_value(self, m):
        return m
