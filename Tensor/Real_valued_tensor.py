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

    def compute_entry_value(self, entry):
        ui = np.ones((self.D,))
        ndim = len(self.dims)

        for dim in range(ndim):
            row_num = entry[dim]
            ui = np.multiply(ui, self.matrices[dim][row_num, :])

        m = np.sum(ui)
        if self.noise != 0:
            s = np.random.normal(0, self.noise)
        else:
            s = 0

        return m + s

    def actual_value(self, m):
        return m


