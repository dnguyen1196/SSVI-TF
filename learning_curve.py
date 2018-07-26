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

default_params = {"mean_update" : "S", "cov_update" : "N", "rank" : 20, "k1" : 64, "k2" : 64}

def get_factorizer_param(model, datatype, diag):
    if model == "deterministic" or model == "simple":
        return  {"eta" : 1, "cov_eta" : 1}

    # TODO: what about robust and diag? do I need sigma_eta
    if model == "robust":
        return  {"eta" : 1, "cov_eta" : 1 if datatype == "real" else 0.001, "sigma_eta" : 0.01, \
                 "unstable_cov" : (True if not diag else False) }

    return None

def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    else:
        mean0 = np.zeros((D,))
    return {"cov0" : cov0, "mean0" : mean0}

def synthesize_tensor(datatype, noise, noise_ratio):
    dims = [50, 50, 50]
    real_dim = 100
    means = [np.ones((real_dim,)) * 5, np.ones((real_dim,)) * 10, np.ones((real_dim,)) * 2]
    covariances = [np.eye(real_dim) * 2, np.eye(real_dim) * 3, np.eye(real_dim) * 2]

    if datatype == "binary":
        tensor = binary_tensor()
    elif datatype == "real":
        tensor = RV_tensor()
    elif datatype == "count":
        tensor = count_tensor()

    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=noise, noise_ratio=noise_ratio)
    return tensor

def do_learning_curve(factorizer, tensor, iter_num):
    train_curve = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    for size in train_curve:
        print("Using ", size, " of training data ")
        tensor.reduce_train_size(size)
        factorizer.reset()
        factorizer.factorize(report=500, max_iteration=iter_num)

def test_learning_curve(datatype, model, diag, noise, iter_num, noise_ratio):
    model = "robust"
    datatype = "real"
    D = 20

    synthetic_tensor = synthesize_tensor(datatype, noise, noise_ratio)
    factorizer_param = get_factorizer_param(model, datatype)
    init_vals = get_init_values(datatype, D)
    params = {**default_params, **factorizer_param, **init_vals, "tensor": synthetic_tensor}

    diag = False  # full or diagonal covariance
    params["diag"] = diag

    if model == "deterministic":
        factorizer = SSVI_TF_d(**params)
    elif model == "simple":
        factorizer = SSVI_TF_simple(**params)
    elif model == "robust":
        factorizer = SSVI_TF_robust(**params)

    do_learning_curve(factorizer, tensor, iter_num)


parser = argparse.ArgumentParser(description='3D tensor factorization synthetic data')
parser.add_argument("-d", "--data", type=str, help="data types: binary, real or count")
parser.add_argument("-m", "--model", type=str, help="model: simple, deterministic or robust")
parser.add_argument("--diag", action="store_true")
parser.add_argument("-n", "--noise", type=float, help="noise level")
parser.add_argument("-i", "--iter", type=int, help="number of iterations")
parser.add_argument("-r", "--ratio", action="store_true")

args = parser.parse_args()

datatype = args.data
model    = args.model
diag     = args.diag
noise    = args.noise
iter_num = args.iter
noise_ratio = args.ratio

if noise is None:
    noise = 0.1

if iter_num is None:
    iter_num = 8000

assert (datatype in ["binary", "real", "count"])
assert (model in ["simple", "deterministic", "robust"])

test_learning_curve(datatype, model, diag, noise, iter_num, noise_ratio)