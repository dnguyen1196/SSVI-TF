import argparse
from SSVI.SSVI_TF_d import SSVI_TF_d
from SSVI.SSVI_TF_simple import SSVI_TF_simple
from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.count_tensor import count_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.real_tensor import RV_tensor
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, help="tensor filename")
parser.add_argument("-d", "--datatype", type=str, help="tensor data type")
parser.add_argument("--diag", action="store_true", help="using diagonal covariance")
parser.add_argument("-re", "--report", type=int, help="report interval", default=10)
parser.add_argument("-iter","--iteration", type=int, help="number of iterations", default=500)
parser.add_argument("-m", "--model", type=str, help="model being used", choices=["deterministic", "simple", "robust"])

args = parser.parse_args()


filename = args.filename
datatype    = args.datatype
model    = args.model

tensor = None
if datatype == "count":
    tensor = count_tensor()
    tensor.read_from_file(filename, 0.7, 0.1, 0.2)
    print("min_count:", tensor.min_count)
    print("max_count:", tensor.max_count)

D = 50

default_params = {"mean_update" : "S", "cov_update" : "N", "rank" : D, "k1" : 64, "k2" : 64}

def get_factorizer_param(model, datatype, diag, using_quadrature):
    set_params = {"eta": 1, "cov_eta":1}

    if datatype != "real":
        if diag:
            set_params["cov_eta"] = 0.1
        if not diag and model == "robust":
            set_params["cov_eta"] = 0.001

    return set_params

def get_init_values(datatype, D):
    cov0 = np.eye(D)
    if datatype == "real":
        mean0 = np.ones((D,))
    else:
        mean0 = np.zeros((D,))
    return {"cov0" : cov0, "mean0" : mean0}


# TODO: to replace by command line option
diag = args.diag
using_quadrature = False

factorizer_param = get_factorizer_param(model, datatype, diag, using_quadrature)
init_vals        = get_init_values(datatype, D)
params           = {**default_params, **factorizer_param, **init_vals, "tensor" : tensor }

params["diag"] = diag
params["batch_size"] = 512
params["cov_eta"] = 1
params["eta"] = 1

if model == "deterministic":
    factorizer = SSVI_TF_d(**params)
elif model == "simple":
    factorizer = SSVI_TF_simple(**params)
elif model == "robust":
    factorizer = SSVI_TF_robust(**params)

factorizer.factorize(report=args.report, max_iteration=args.iteration, detailed_report=False)

#v1 = tensor.find_observed_ui(0, 50)
#v2 = tensor.find_observed_ui(1,322)
#v3 = tensor.find_observed_ui(2, 1000)
#v4 = tensor.find_observed_ui(3, 2)

#print(v1)
