import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.binary_tensor import binary_tensor

np.random.seed(seed=317) # For control and comparisons

# Generate synthesize tensor, true, this is what we try to recover
dims        = [50, 50, 50]
hidden_D    = 20
means       = [np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0]
covariances = [np.eye(hidden_D)*0.1, np.eye(hidden_D), np.eye(hidden_D)*0.5]

data        = binary_tensor()

data.synthesize_data(dims, means, covariances, hidden_D, 0.8, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20

# Approximate posterior initial guess
mean0 = np.zeros((D,))
cov0 = np.eye(D)

############################### FACTORIZATION ##########################
mean_update = "S"
cov_update  = "N"

factorizer = SSVI_TF_robust(data, rank=D, \
                            mean_update=mean_update, cov_update=cov_update, \
                            k1=64, k2=64,\
                            eta=1, cov_eta=0.001,\
                            mean0=mean0, cov0=cov0)

factorizer.factorize(report=50)
