import numpy as np

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF_d import H_SSVI_TF_2d
from Tensor.Tensor import tensor

# np.random.seed(seed=317) # For control and comparisons
# Generate synthesize tensor, true, this is what we try to recover

dims     = [50, 50, 50]
hidden_D = 20

data = tensor(datatype="count")
data.synthesize_count_data(dims, hidden_D, 0.8, 0.1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension

D = 20
p_likelihood = distribution("poisson", 1, None, None)

approximate_mean = np.ones((D,)) * 0.1
approximate_cov = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean, approximate_cov))

# Model prior
m = np.ones((D,))
S = np.eye(D)
p_prior = distribution("normal", 1, ("approximate_mean_0", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior)

############################### FACTORIZATION ##########################

rho_cov = lambda t: 0.01
factorizer = H_SSVI_TF_2d(model, data, rank=D, rho_cov=rho_cov)
factorizer.factorize(report=1000)
