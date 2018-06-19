import numpy as np

from Model.TF_Models import Posterior_Full_Covariance
from SSVI.SSVI_TF_d import SSVI_TF_d
from Tensor.Tensor import Tensor

# np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover

dims     = [10, 10]
hidden_D = 20

data = Tensor(datatype="count")
data.synthesize_count_data(dims, hidden_D, 0.8, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension

D = 20

mean0 = np.ones((D,)) * 0.1
cov0 = np.eye(D)

############################### FACTORIZATION ##########################

mean_update = "S"
cov_update  = "N"
factorizer = SSVI_TF_d(data, rank=D, \
                                   mean_update=mean_update, cov_update=cov_update, \
                                   mean0=mean0, cov0=cov0)

factorizer.factorize(report=50)
