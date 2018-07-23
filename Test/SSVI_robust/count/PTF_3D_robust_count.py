import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.count_tensor import count_tensor

np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover

dims     = [20, 20, 20]
hidden_D = 20
means       = [np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0]
covariances = [np.eye(hidden_D) * 0.1, np.eye(hidden_D), np.eye(hidden_D)*0.5]

tensor = count_tensor()
tensor.synthesize_data(dims, means, covariances, hidden_D, 0.8, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension

D = 20

mean0 = np.ones((D,))
cov0 = np.eye(D)

############################### FACTORIZATION ##########################

mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_robust(tensor, rank=D, \
                                   mean_update=mean_update, cov_update=cov_update, diag=False,\
                                   k1=64, k2=64,\
                                   eta=1, cov_eta=0.001, \
                                   mean0=mean0, cov0=cov0)

factorizer.factorize(report=100)
