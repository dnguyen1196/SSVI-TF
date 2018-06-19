import numpy as np

from Model.TF_Models import ApproximatePosteriorParams
from SSVI.SSVI_TF_deterministic import SSVI_TF_deterministic
from Tensor.Tensor import Tensor


# Generate synthesize tensor, true, this is what we try to recover
dims        = [50, 50, 50]
hidden_D    = 20
means       = [np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0]
covariances = [np.eye(hidden_D)*0.1, np.eye(hidden_D), np.eye(hidden_D)*0.5]
data        = Tensor(datatype="binary")

data.synthesize_binary_data(dims, hidden_D, 0.5, 0.1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20

# Approximate posterior initial guess
mean0 = np.ones((D,)) * 0
cov0 = np.eye(D)

############################### FACTORIZATION ##########################
mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_deterministic(data, rank=D, \
                                   mean_update=mean_update, cov_update=cov_update, \
                                   mean0=mean0, cov0=cov0)

factorizer.factorize(report=100)
