import numpy as np

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF_d import H_SSVI_TF_2d
from Tensor.Tensor import tensor

# np.random.seed(seed=317) # For control and comparisons

# Generate synthesize tensor, true, this is what we try to recover
dims     = [100, 100, 100]
hidden_D = 20
means    = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 10, np.ones((hidden_D,)) * 2]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]
data = tensor()
data.synthesize(dims, means, covariances, hidden_D, 0.8, 0.1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20
# Likelihood function for real valued tensor
p_likelihood = distribution("normal", 1, ("var"), 1)

# likelihood funcion for binary valued tensor
# p_likelihood = distribution("bernoulli", 1, None, None)
# likelihood function for count-valued tensor
# Approximate posterior initial

approximate_mean = np.ones((D,)) * 5
approximate_cov = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean, approximate_cov))

# Model prior
m = np.ones((D,))
S = np.eye(D)
p_prior = distribution("normal", 1, ("approximate_mean_0", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior)

############################### FACTORIZATION ##########################

rho_cov = lambda t: 0.01
factorizer = H_SSVI_TF_2d(model, data, rank=D, rho_cov=rho_cov, scheme="adagrad")
factorizer.factorize()
