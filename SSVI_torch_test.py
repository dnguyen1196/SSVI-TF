from Baseline.SSVI_pytorch import SSVI_torch
from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np

# np.random.seed(42)

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
    # tensor.reduce_train_size(0.1)
    return tensor


# test_tensor = synthesize_tensor([50,50, 50], "real", False, 0)
# test_tensor.reduce_train_size(0.05)

test_tensor = synthesize_tensor([20, 20, 20], "real", True, 0.05)
# test_tensor.reduce_train_size(0.05)

gradient_update = "S"
rank = 50

factorizer = SSVI_torch(test_tensor, gradient_update=gradient_update, rank=rank)

# factorizer.factorize(10000, algorithm="SGD", lr=0.0001, report=[0,1,2], interval=10)

# factorizer.factorize(10000, algorithm="Adam", lr=0.1, report=[0,1,2], interval=10)

# factorizer.factorize(10000, algorithm="RMSProp", lr=1, report=[0,1,2], interval=10)

factorizer.factorize(10000, algorithm="Adagrad", lr=1, report=[0,1,2], interval=10)
