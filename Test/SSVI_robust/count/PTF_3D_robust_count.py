import numpy as np

from SSVI.SSVI_TF_robust import SSVI_TF_robust
from Tensor.Tensor import Tensor

np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover

dims     = [20, 20, 20]
hidden_D = 20

tensor = Tensor(datatype="count")
tensor.synthesize_count_data(dims, hidden_D, 0.8, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension

D = 20

mean0 = np.ones((D,))
cov0 = np.eye(D)

############################### FACTORIZATION ##########################

mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_robust(tensor, rank=D, \
                                   mean_update=mean_update, cov_update=cov_update, \
                                   k1=32, k2=32,\
                                   eta=1, cov_eta=0.01,\
                                   mean0=mean0, cov0=cov0)
factorizer.factorize(report=50)
