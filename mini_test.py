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
    set_params = {"eta" : 1, "cov_eta": 0.001}
    # Keep it consistent across all models
    return set_params

def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    elif datatype == "binary":
        mean0 = np.zeros((D,))
    else:
        mean0 = np.ones((D,)) * 0.1
    return {"cov0" : cov0, "mean0" : mean0}

def synthesize_tensor(dims, datatype, using_ratio, noise):
    #dims = [50, 50, 50]
    #dims = [25, 25, 25]
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

def synthesize_matrix(dims, datatype, noise_ratio, noise_amount):
    real_dim = 100
    means = [np.ones((real_dim,)) * 5, np.ones((real_dim,)) * 2]
    covariances = [np.eye(real_dim) * 2, np.eye(real_dim) * 3]

    if datatype == "binary":
        tensor = binary_tensor()
    elif datatype == "real":
        tensor = RV_tensor()
    else:
        tensor = count_tensor()

    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=noise_amount, noise_ratio=noise_ratio)
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
parser.add_argument("--matrix", action="store_true", help="Doing matrix factorization instead of tensor factorization")
parser.add_argument("-ceta", "--cov_eta", type=float, help="cov eta", default=1.0)
parser.add_argument("--rand", action="store_true", help="Using random start")
parser.add_argument("-meta", "--mean_eta", type=float, help="mean eta", default=1.0)
parser.add_argument("-dim", "--dimension", nargs='+', required=True)

args = parser.parse_args()

model    = args.model
datatype = args.datatype

D        = 20
diag = args.diag # full or diagonal covariance
NOISE_RATIO = args.ratio # using noise as ratio of f
NOISE_AMOUNT = args.noise # Noise amount
# Should be exclusive group

default_params["diag"] = diag
fixed_covariance = args.fixed_cov
using_quadrature = args.quadrature
randstart = args.rand
dims      = [int(x) for x in args.dimension]

if NOISE_RATIO is not None:
    using_ratio = True
    noise = args.ratio
elif NOISE_AMOUNT is not None:
    using_ratio = False
    noise = args.noise
else:
    using_ratio = True
    noise = 0

if len(dims) == 3:
    synthetic_tensor = synthesize_tensor(dims, datatype, using_ratio, noise)
elif len(dims) == 2:
    synthetic_tensor = synthesize_matrix(dims, datatype, using_ratio, noise)
else:
    raise Exception("Have not implemented the necessary dimensions")

factorizer_param = get_factorizer_param(model, datatype, diag, using_quadrature)
init_vals        = get_init_values(datatype, D)

params           = {**default_params, **factorizer_param, **init_vals, "tensor" : synthetic_tensor }

params["cov_eta"] = args.cov_eta
params["eta"] = args.mean_eta
params["randstart"] = randstart

params["cov0"] = np.eye(D)



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

max_iterations = args.num_iters
#if datatype == "count" and model != "robust" and args.matrix:
#    max_iterations = 30000


factorizer.factorize(report=args.report, max_iteration=max_iterations, fixed_covariance=fixed_covariance, to_report=[0, 5, 10, 20,  50, 100, 200])

v1, _ = factorizer.posterior.get_vector_distribution(0, 1)
v2, _ = factorizer.posterior.get_vector_distribution(1, 4)
v3, _ = factorizer.posterior.get_vector_distribution(2, 3)
print(v1)
print(v2)
print(v3)
