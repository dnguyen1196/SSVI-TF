import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.Tensor import Tensor

np.random.seed(seed=317)
# For control and comparison
# Generate synthesize data, true, this is what we try to recover

dims     = [50, 50, 50]
# dims = [10,10,10]
hidden_D = 20
means    = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 10, np.ones((hidden_D,)) * 2]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]
data = Tensor()

# data.synthesize_real_data(dims, means, covariances, hidden_D, 0.8, 1)
data.synthesize_real_data(dims, means, covariances, hidden_D, 0.2, 1, noise=1)

################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20
mean0 = np.ones((D,)) * 5
cov0 = np.eye(D)

mean_update = "S"
cov_update  = "N"

factorizer  = SSVI_TF_robust(data, rank=D, \
                            mean_update=mean_update, cov_update=cov_update, \
                            mean0=mean0, cov0=cov0, k1=128, k2=64, \
                            eta=1, cov_eta=0.1, sigma_eta=1)

factorizer.factorize(report=50)
