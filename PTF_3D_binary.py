import numpy as np

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF_d import H_SSVI_TF_2d
from Tensor.Tensor import tensor


# Generate synthesize tensor, true, this is what we try to recover
dims        = [50, 50, 50] # 100 * 100 * 100 tensor
hidden_D    = 10
means       = [np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0, np.ones((hidden_D,)) * 0]
covariances = [np.eye(hidden_D)*0.1, np.eye(hidden_D), np.eye(hidden_D)*0.5]
data        = tensor(datatype="binary")

data.synthesize_binary_data(dims, hidden_D, 0.5, 0.1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 10
p_likelihood = distribution("bernoulli", 1, None, None)

# Approximate posterior initial guess
approximate_mean_0 = np.ones((D,)) * 0
approximate_cov_0 = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean_0, approximate_cov_0))

# Model prior
m = np.zeros((D,))
S = np.eye(D)
p_prior = distribution("normal", 1, ("mean", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior, likelihood_type="bernoulli")

############################### FACTORIZATION ##########################
rho_cov = lambda t : 0.01
factorizer = H_SSVI_TF_2d(model, data, rank=D, rho_cov=rho_cov, k1=10, k2=10)
factorizer.factorize()
