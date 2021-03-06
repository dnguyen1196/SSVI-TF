from Baseline.CP_ALS import CPDecompPytorch3D
from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np

np.random.seed(42)

def synthesize_tensor(dims, datatype, using_ratio, noise):
    real_dim = 100
    means = [np.ones((real_dim,)) * 5, np.ones((real_dim,)) * 10, np.ones((real_dim,)) * 2]
    covariances = [np.eye(real_dim) * 2, np.eye(real_dim) * 3, np.eye(real_dim) * 2]

    if datatype == "binary":
        tensor = binary_tensor()
    elif datatype == "real":
        tensor = RV_tensor()
    elif datatype == "count":
        tensor = count_tensor()

    """
    """
    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=noise, noise_ratio=using_ratio)
    return tensor


test_tensor = synthesize_tensor([20,20,20], "count", False, 0)

factorizer = CPDecompPytorch3D(test_tensor, rank=10)
factorizer.factorize(10000)