import numpy as np

from SSVI.SSVI_TF_d import SSVI_TF_d
from Tensor.count_tensor import count_tensor


def do_learning_curve_real(factorizer, tensor):
    train_curve = [0.2, 0.4, 0.6, 0.8, 1]

    for size in train_curve:
        print("Using ", size, " of training data ")
        tensor.reduce_train_size(size)
        factorizer.reset()
        factorizer.factorize(report=50, max_iteration=2000)


"""
Generating tensor
"""
dims = [50, 50, 50]
D = 20
means = [np.ones((D,)) * 5, np.ones((D,)) * 10, np.ones((D,)) * 2]
covariances = [np.eye(D) * 2, np.eye(D) * 3, np.eye(D) * 2]

data = count_tensor()
data.synthesize_data(dims, means, covariances, D)

"""
Initializing factorizer
"""
fact_D = 20
mean0 = np.zeros((D,))
cov0 = np.eye(D)

mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_d(data, rank=fact_D, \
                       mean_update=mean_update, cov_update=cov_update, \
                       k1=64, k2=64, \
                       mean0=mean0, cov0=cov0)

do_learning_curve_real(factorizer, data)
# factorizer.factorize(report=10, max_iteration=2000)
