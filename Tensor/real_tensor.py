from Tensor.Tensor import Tensor
import numpy as np

class RV_tensor(Tensor):
    def __init__(self):
        super(RV_tensor, self).__init__(datatype="real")
        self.link_fun = lambda m: m

    def generate_hidden_matrices(self, means, covariances):
        ndim = len(self.dims)
        matrices = [[] for _ in range(ndim)]
        # Generate the random hidden matrices
        for i in range(ndim):
            matrices[i] = self.create_random_matrix(self.dims[i], self.D, means[i], covariances[i])
        return matrices

    def data_link_fun(self, m):
        return m

    def actual_value(self, m):
        return m


