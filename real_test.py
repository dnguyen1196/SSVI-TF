import sys
import os
import argparse

from Tensor.real_tensor import RV_tensor
from Tensor.binary_tensor import binary_tensor
from Tensor.count_tensor import count_tensor
import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from SSVI.SSVI_TF_d import SSVI_TF_d
from SSVI.SSVI_TF_simple import SSVI_TF_simple



parser = argparse.ArgumentParser(description="Real life data set testing")
parser.add_argument("-m", "--model", type=str, help="model of factorizer", choices=["deterministic", "simple", "robust"], default="simple")
parser.add_argument("-d", "--datatype", type=str, help="datatype of tensor", choices=["real", "binary", "count"], required=True)
parser.add_argument("--diag", action="store_true")
parser.add_argument("-f", "--filename", type=str, help="tensor data type", required=True)
parser.add_argument("-it", "--num_iters", type=int, help="Max number of iterations", default=8000)
parser.add_argument("-re", "--report", type=int, help="Report interval", default=500)
parser.add_argument("-ceta", "--cov_eta", type=float, help="cov eta", default=1.0)
parser.add_argument("-meta", "--mean_eta", type=float, help="mean eta", default=1.0)
parser.add_argument("-k1", "--k1", type=int, help="k1 samples", default=64)
parser.add_argument("-k2", "--k2", type=int, help="k2 samples", default=128)
parser.add_argument("-R", "--rank", type=int, help="factorization rank", default=50)
parser.add_argument("-o", "--output", type=str, help="Output folder", default="learning_results")


args = parser.parse_args()

model = args.model
datatype = args.datatype
filename = args.filename
diag = args.diag
num_iters = args.num_iters
report = args.report
m_eta = args.mean_eta
c_eta = args.cov_eta
k1 = args.k1
k2 = args.k2
D = args.rank
output_folder = args.output



tensor = None

if datatype == "real":
    tensor = RV_tensor()
elif datatype == "binary":
    tensor = binary_tensor()
else:
    tensor = count_tensor()

# Read from file to initialize tensor
tensor.read_from_file(filename, 0.8, 0, 0.2)

# Initialize the parameters for the algorithm
default_params = {"mean_update" : "S", "cov_update" : "N", "rank" : D}
factorizer_param = {"eta" : 1, "cov_eta": 0.001}

params            = {**default_params, **factorizer_param, "tensor" : tensor }
params["cov_eta"] = args.cov_eta
params["eta"]     = args.mean_eta
params["k1"]      = args.k1
params["k2"]      = args.k2
params["diag"]    = diag

# Try different mean0 initialization
params["mean0"]  = np.zeros((D,))

if model == "deterministic":
    factorizer = SSVI_TF_d(**params)
elif model == "simple":
    factorizer = SSVI_TF_simple(**params)
elif model == "robust":
    factorizer = SSVI_TF_robust(**params)



factorizer.factorize(report=report, max_iteration=num_iters, \
    to_report=[0, 5, 10, 20,  50, 100, 200], detailed_report=False, output_folder=output_folder)


