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

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

np.random.seed(seed=319)

default_params = {"mean_update" : "S", "cov_update" : "N", "rank" : 20, "k1" : 64, "k2" : 64, "eta":1, "cov_eta":1, "randstart":True}


def get_factorizer_param(model, datatype, diag):
    return default_params


def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    else:
        mean0 = np.zeros((D,))
    return {"cov0" : cov0, "mean0" : mean0}


def synthesize_tensor(datatype, dims, noise, noise_ratio):
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


def do_learning_curve(factorizer,  tensor, iter_num):
    # Perform learning curve by exposing the algorithm to varying percentage
    # of training data
    train_curve = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    for size in train_curve:
        print("Using ", size, " of training data ")
        tensor.reduce_train_size(size)
        factorizer.reset()
        factorizer.factorize(report=500, max_iteration=iter_num, to_report=[0, 10, 25])


def test_learning_curve(datatype, model, diag, noise, iter_num, noise_ratio, dims):
    # The first step is to synthesize the tensor
    # according to the given dimension and datatype and noise
    if noise_ratio is not None:
        noise_ratio = True
        noise = noise_ratio
    else:
        noise_ratio = False
    tensor = synthesize_tensor(datatype, dims, noise, noise_ratio)

    # Next step is to initialize the factorizer with appropriate
    # parameters
    D = 20
    init_vals = get_init_values(datatype, D)
    params = {**default_params, **init_vals, "tensor": tensor}
    params["diag"] = diag

    if model == "deterministic":
        factorizer = SSVI_TF_d(**params)
    elif model == "simple":
        factorizer = SSVI_TF_simple(**params)
    elif model == "robust":
        factorizer = SSVI_TF_robust(**params)

    # Evaluate the 'true' model
    factorizer.evaluate_true_params()

    # Perform learning curve
    do_learning_curve(factorizer, tensor, iter_num)



parser = argparse.ArgumentParser(description="Testing models at specific training size")
parser.add_argument("-m", "--model", type=str, help="model of factorizer", choices=["deterministic", "simple", "robust"])
parser.add_argument("-d", "--datatype", type=str, help="datatype of tensor", choices=["real", "binary", "count"])

excl_group = parser.add_mutually_exclusive_group()
excl_group.add_argument("-r", "--ratio", type=float, help="noise as ratio")
excl_group.add_argument("-n", "--noise", type=float, help="noise level", default=0.0)

parser.add_argument("--diag", action="store_true")
parser.add_argument("-tr", "--train_size", type=float, help="portion of training data")
parser.add_argument("--fixed_cov", action="store_true", help="Fixed covariance")
parser.add_argument("-it", "--num_iters", type=int, help="Max number of iterations", default=10000)
parser.add_argument("-re", "--report", type=int, help="Report interval", default=500)
parser.add_argument("--quadrature", action="store_true", help="using quadrature")
parser.add_argument("--matrix", action="store_true", help="Doing matrix factorization instead of tensor factorization")
parser.add_argument("-ceta", "--cov_eta", type=float, help="cov eta", default=1.0)
parser.add_argument("--rand", action="store_true", help="Using random start")
parser.add_argument("-meta", "--mean_eta", type=float, help="mean eta", default=1.0)
parser.add_argument("-dim", "--dimension", nargs='+', required=True, default=[50, 50, 50])
parser.add_argument("-k1", "--k1", type=int, help="k1 samples", default=64)
parser.add_argument("-k2", "--k2", type=int, help="k2 samples", default=128)

args = parser.parse_args()

datatype = args.datatype
model    = args.model
diag     = args.diag
noise    = args.noise
iter_num = args.num_iters
noise_ratio = args.ratio
dims = [int(x) for x in args.dimension]

test_learning_curve(datatype, model, diag, noise, iter_num, noise_ratio, dims)


"""
hard noise for real -> 500
hard noise for binary -> 0.5
hard noise for count -> 1
"""
