from Baseline.SSVI_pytorch import SSVI_torch
from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np

np.random.seed(42)

def synthesize_tensor(dims, datatype, using_ratio, noise):
    real_dim = 50
    means = [np.ones((real_dim,)) * 1, np.ones((real_dim,)) * 4, np.ones((real_dim,)) * 2]
    covariances = [np.eye(real_dim) * 0.5, np.eye(real_dim) * 0.5, np.eye(real_dim) * 0.5]

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
    # tensor.reduce_train_size(0.1)
    return tensor


test_tensor = synthesize_tensor([20,20], "real", False, 0)
# test_tensor.reduce_train_size()
natural_gradient = "H"
factorizer = SSVI_torch(test_tensor, using_natural_gradient=natural_gradient, rank=10)

factorizer.factorize(10000, algorithm="AdaGrad", lr=1, report=[0,1,2,3,4,5,6], interval=10)