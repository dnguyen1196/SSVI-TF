import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np

from Model.TF_Models import Posterior_Full_Covariance
from SSVI.SSVI_TF_d import SSVI_TF_d
from Tensor.Tensor import Tensor

np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover
dims     = [20, 20, 20]
hidden_D = 20
means    = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 10, np.ones((hidden_D,)) * 2]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]
tensor = Tensor()
tensor.synthesize_real_data(dims, means, covariances, hidden_D, 0.8, 0.1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20
mean0 = np.ones((D,)) * 5
cov0  = np.eye(D)

############################### FACTORIZATION ##########################
mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_d(tensor, rank=D, \
                           mean_update=mean_update, cov_update=cov_update, \
                           mean0=mean0, cov0=cov0)
factorizer.factorize(report=10)
