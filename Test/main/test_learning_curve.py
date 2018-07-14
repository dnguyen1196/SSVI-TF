import sys
import pickle
import argparse
from Tensor.Real_valued_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor

import numpy as np


# TODO: function to do learning curve
def do_learning_curve():
    return

real_tensor = RV_tensor()
binary_tensor = binary_tensor()
count_tensor  = count_tensor()


dims     = [50, 50, 50]
D        = 20
means    = [np.ones((D,)) * 5, np.ones((D,)) * 10, np.ones((D,)) * 2]
covariances = [np.eye(D) * 2, np.eye(D) * 3, np.eye(D) * 2]


real_tensor.synthesize_data(dims, means, covariances, D)
real_tensor.reduce_train_size(0.8)

binary_tensor.synthesize_data(dims, means, covariances)
binary_tensor.reduce_train_size(0.8)

count_tensor.synthesize_data(dims, means, covariances)
count_tensor.reduce_train_size(0.8)

# TODO: test on real data sets