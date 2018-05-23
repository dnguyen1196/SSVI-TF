import numpy as np

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF_d import H_SSVI_TF_2d
from Tensor.Tensor import tensor


# Generate synthesize tensor, true, this is what we try to recover
dims        = [50, 50, 50] # 100 * 100 * 100 tensor
hidden_D    = 5
means       = [np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0.]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]
data        = tensor(datatype="binary")

data.synthesize(dims, means, covariances, hidden_D, 0.5, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 5
p_likelihood = distribution("bernoulli", 1, None, None)

# Approximate posterior initial guess
approximate_mean = np.ones((D,))
approximate_cov = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean, approximate_cov))

# Model prior
m = np.ones((D,)) * 0.5
S = np.eye(D)
p_prior = distribution("normal", 1, ("approximate_mean", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior, likelihood_type="bernoulli")

############################### FACTORIZATION ##########################
rho_cov = lambda t : 1
factorizer = H_SSVI_TF_2d(model, data, rank=D, rho_cov=rho_cov, k1=10, k2=10, scheme="sgd")
factorizer.factorize()
