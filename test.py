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
<<<<<<< HEAD
        return  {"eta" : 5, "cov_eta" : 1 if datatype == "real" else 0.001, "sigma_eta" : 0.01, \
=======
        cov_eta = 1
        eta     = 1
        if datatype == "count":
            cov_eta = 0.001
            eta     = 1
        elif datatype == "binary":
            cov_eta = 0.001
            eta     = 1

        return  {"eta" : eta, "cov_eta" : cov_eta, "sigma_eta" : 0.1, \
>>>>>>> 23eca854e7d197aebd8f7cba1e06328f4928f4f5
                 "unstable_cov" : (True if not diag else False) }

    return None

def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    else:
        mean0 = np.zeros((D,))
    return {"cov0" : cov0, "mean0" : mean0}

def synthesize_tensor(datatype, NOISE_RATIO):
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

    NOISE = 0.1
    if NOISE_RATIO:
        if datatype == "real":
            NOISE = 500
        elif datatype == "binary":
            NOISE = 0.5
        else:
            NOISE = 0.1

    tensor.synthesize_data(dims, means, covariances, real_dim, \
                           train=0.8, sparsity=1, noise=NOISE, noise_ratio=NOISE_RATIO)
    return tensor


<<<<<<< HEAD
model    = "robust"
=======
model    = "simple"
>>>>>>> 23eca854e7d197aebd8f7cba1e06328f4928f4f5
datatype = "binary"
D        = 20
diag = False # full or diagonal covariance
NOISE_RATIO = False
train_size  = 0.05
default_params["diag"] = diag
fixed_covariance = True


synthetic_tensor = synthesize_tensor(datatype, NOISE_RATIO)
factorizer_param = get_factorizer_param(model, datatype, diag)
init_vals        = get_init_values(datatype, D)
params           = {**default_params, **factorizer_param, **init_vals, "tensor" : synthetic_tensor }

<<<<<<< HEAD
params["eta"]    = 1
params["sigma_eta"]  = 1
=======
if fixed_covariance: # Special option to test, keep a fixed covariance
    if datatype == "binary" or datatype == "count":
        params["cov0"] = np.eye(D) * 0.1

>>>>>>> 23eca854e7d197aebd8f7cba1e06328f4928f4f5

if model == "deterministic":
    factorizer = SSVI_TF_d(**params)
elif model == "simple":
    factorizer = SSVI_TF_simple(**params)
elif model == "robust":
    factorizer = SSVI_TF_robust(**params)

<<<<<<< HEAD
synthetic_tensor.reduce_train_size(train_size)
factorizer.evaluate_true_params()
factorizer.factorize(report=100, max_iteration=2000)

=======
portion = 0.05

factorizer.evaluate_true_params()

synthetic_tensor.reduce_train_size(portion)
factorizer.factorize(report=500, max_iteration=6000, fixed_covariance=fixed_covariance)
>>>>>>> 23eca854e7d197aebd8f7cba1e06328f4928f4f5
