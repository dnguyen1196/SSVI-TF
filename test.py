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

def get_factorizer_param(model, datatype, diag, using_quadrature):
    if model == "deterministic" or model == "simple":
        return  {"eta" : 1, "cov_eta" : 1}

    # TODO: what about robust and diag? do I need sigma_eta
    if model == "robust":
        hermite_quad = using_quadrature
        cov_eta = 1
        eta     = 1
        if datatype == "count":
            cov_eta = 0.001
            eta     = 1
        elif datatype == "binary":
            if diag:
                cov_eta = 0.00001
            else:
                cov_eta = 0.001
            eta     = 1

        return  {"eta" : eta, "cov_eta" : cov_eta, "sigma_eta" : 1, \
                 "unstable_cov" : (True if not diag else False), "quadrature" : hermite_quad }

    return None

def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    else:
        mean0 = np.zeros((D,))
    return {"cov0" : cov0, "mean0" : mean0}

def synthesize_tensor(datatype, NOISE_RATIO, NOISE_AMOUNT):
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

    """
    NOISE = 0.1
    if not NOISE_RATIO:
        if datatype == "real":
            NOISE = 500
        elif datatype == "binary":
            NOISE = 0.5
        else:
            NOISE = 0.1
    """
    if NOISE_AMOUNT is not None:
        using_ratio = False
        noise = NOISE_AMOUNT
    elif NOISE_RATIO is not None:
        using_ratio = True
        noise = NOISE_RATIO
    else:
        using_ratio = True
        noise = 0

    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=noise, noise_ratio=using_ratio)
    return tensor


parser = argparse.ArgumentParser(description="Testing models at specific training size")
parser.add_argument("-m", "--model", type=str, help="model of factorizer", choices=["deterministic", "simple", "robust"])
parser.add_argument("-d", "--datatype", type=str, help="datatype of tensor", choices=["real", "binary", "count"])

excl_group = parser.add_mutually_exclusive_group()
excl_group.add_argument("-r", "--ratio", type=float, help="noise as ratio")
excl_group.add_argument("-n", "--noise", type=float, help="noise level")

parser.add_argument("--diag", action="store_true")
parser.add_argument("-tr", "--train_size", type=float, help="portion of training data")
parser.add_argument("--fixed_cov", action="store_true", help="Fixed covariance")
parser.add_argument("-it", "--num_iters", type=int, help="Max number of iterations", default=8000)
parser.add_argument("-re", "--report", type=int, help="Report interval", default=500)
parser.add_argument("--quadrature", action="store_true", help="using quadrature")

args = parser.parse_args()

model    = args.model
datatype = args.datatype

D        = 20
diag = args.diag # full or diagonal covariance
NOISE_RATIO = args.ratio # using noise as ratio of f?
NOISE_AMOUNT = args.noise # Noise amount # Should be exclusive group

default_params["diag"] = diag
fixed_covariance = args.fixed_cov
using_quadrature = args.quadrature

synthetic_tensor = synthesize_tensor(datatype, NOISE_RATIO, NOISE_AMOUNT)
factorizer_param = get_factorizer_param(model, datatype, diag, using_quadrature)
init_vals        = get_init_values(datatype, D)
params           = {**default_params, **factorizer_param, **init_vals, "tensor" : synthetic_tensor }


if fixed_covariance: # Special option to test, keep a fixed covariance
    if datatype == "binary" or datatype == "count":
        params["cov0"] = np.eye(D) * 0.1


if model == "deterministic":
    factorizer = SSVI_TF_d(**params)
elif model == "simple":
    factorizer = SSVI_TF_simple(**params)
elif model == "robust":
    factorizer = SSVI_TF_robust(**params)


portion = args.train_size

factorizer.evaluate_true_params()

if portion is not None:
    synthetic_tensor.reduce_train_size(portion)

factorizer.factorize(report=args.report, max_iteration=args.num_iters, fixed_covariance=fixed_covariance, to_report=[50, 100, 200])

v1, _ = factorizer.posterior.get_vector_distribution(0, 1)
v2, _ = factorizer.posterior.get_vector_distribution(1, 10)
v3, _ = factorizer.posterior.get_vector_distribution(2, 30)
print(v1)
print(v2)
print(v3)

