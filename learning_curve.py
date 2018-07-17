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

np.random.seed(seed=319)

def do_learning_curve(factorizer, tensor):
    train_curve = [0.2, 0.4, 0.6, 0.8, 1]
    for size in train_curve:
        print("Using ", size, " of training data ")
        tensor.reduce_train_size(size)
        factorizer.reset()
        factorizer.factorize(report=50, max_iteration=2000)


def test_learning_curve(datatype, model, diag):
    diag_cov = False
    if diag:
        diag_cov = True

    assert (datatype in ["real", "binary", "count"])
    assert (model in ["deterministic", "simple", "robust"])

    dims = [50, 50, 50]
    hidden_D = 20
    means = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 10, np.ones((hidden_D,)) * 2]
    covariances = [np.eye(hidden_D) * 2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]

    D = 20
    if datatype == "real":
        tensor = RV_tensor()
        mean0 = np.ones((D,)) * 5
        eta = 1
        cov_eta = 0.01

    elif datatype == "binary":
        tensor = binary_tensor()
        mean0 = np.zeros((D,))
        eta = 1
        cov_eta = 0.01

    elif datatype == "count":
        tensor = count_tensor()
        mean0 = np.zeros((D,))
        eta = 1
        cov_eta = 0.01

    tensor.synthesize_data(dims, means, covariances, hidden_D, noise=100)

    mean_update = "S"
    cov_update = "N"
    fact_D = 20
    cov0 = np.eye(D)

    if model == "deterministic":
        factorizer = SSVI_TF_d(tensor, rank=fact_D, \
                               mean_update=mean_update, cov_update=cov_update, diag=diag_cov,\
                               k1=64, k2=64, \
                               mean0=mean0, cov0=cov0)
    elif model == "simple":
        factorizer = SSVI_TF_simple(tensor, rank=fact_D, \
                               mean_update=mean_update, cov_update=cov_update, diag=diag_cov, \
                               k1=64, k2=64, \
                               mean0=mean0, cov0=cov0)

    elif model == "robust":
        factorizer = SSVI_TF_robust(tensor, rank=fact_D, \
                                mean_update=mean_update, cov_update=cov_update, diag=diag_cov, \
                                mean0=mean0, cov0=cov0, k1=64, k2=64, \
                                eta=eta, cov_eta=cov_eta)

    do_learning_curve(factorizer, tensor)

parser = argparse.ArgumentParser(description='3D tensor factorization synthetic data')
parser.add_argument("-d", "--data", type=str, help="data types: binary, real or count")
parser.add_argument("-m", "--model", type=str, help="model: simple, deterministic or robust")
parser.add_argument("--diag", action="store_true")

args = parser.parse_args()

datatype = args.data
model    = args.model
diag     = args.diag

assert (datatype in ["binary", "real", "count"])
assert (model in ["simple", "deterministic", "robust"])

test_learning_curve(datatype, model, diag)