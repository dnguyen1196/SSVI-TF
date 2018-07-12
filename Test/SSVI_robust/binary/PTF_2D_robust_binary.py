import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.Tensor import Tensor

np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover

dims     = [50, 50]
hidden_D = 20

data = Tensor(datatype="binary")
data.synthesize_binary_data(dims, hidden_D, 0.8, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension

D = 20

mean0 = np.ones((D,)) * 0.1
cov0 = np.eye(D)

############################### FACTORIZATION ##########################

mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_robust(data, rank=D, \
                                   mean_update=mean_update, cov_update=cov_update, \
                                   k1=64, k2=64,\
                                   eta=1, cov_eta=0.01,\
                                   mean0=mean0, cov0=cov0)

factorizer.factorize(report=10)
