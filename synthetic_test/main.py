import sys
import pickle
import argparse
from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from SSVI.SSVI_TF_d import SSVI_TF_d
from SSVI.SSVI_TF_simple import SSVI_TF_simple


def do_learning_curve(factorizer, tensor):
    train_curve = [0.2, 0.4, 0.6, 0.8, 1]

    for size in train_curve:
        print("Using ", size, " of training data ")
        tensor.reduce_train_size(size)
        factorizer.reset()
        factorizer.factorize(report=50, max_iteration=2000)


def test_learning_curve(datatype, model):
    assert (datatype in ["real", "binary", "count"])
    assert (model in ["deterministic", "simple", "robust"])
    mean_update = "S"
    cov_update = "N"
    tensor = None
    D = 20
    if datatype == "real":
        tensor = RV_tensor()
        mean0 = np.ones((D,)) * 5
        cov0 = np.eye(D)

    elif datatype == "binary":
        tensor = binary_tensor()
        mean0 = np.zeros((D,))
        cov0 = np.eye(D)

    elif datatype == "count":
        tensor = count_tensor()

    factorizer = None
    fact_D = 20

    if model == "deterministic":
        factorizer = SSVI_TF_d(real_tensor, rank=fact_D, \
                               mean_update=mean_update, cov_update=cov_update, \
                               k1=64, k2=64, \
                               mean0=mean0, cov0=cov0)
    elif model == "simple":
        factorizer = SSVI_TF_simple(data, rank=D, \
                                    mean_update=mean_update, cov_update=cov_update, \
                                    k1=64, k2=64, \
                                    mean0=mean0, cov0=cov0)

    elif model == "robust":
        factorizer = SSVI_TF_robust(data, rank=D, \
                                    mean_update=mean_update, cov_update=cov_update, \
                                    mean0=mean0, cov0=cov0, k1=128, k2=64, \
                                    eta=1, cov_eta=0.1)

    do_learning_curve(factorizer, tensor)


test_learning_curve("real", "simple")