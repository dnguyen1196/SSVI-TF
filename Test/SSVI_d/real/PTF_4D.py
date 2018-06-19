
import numpy as np
# np.random.seed(123)

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF_d import SSVI_TF
from Tensor.Tensor import Tensor


# Generate synthesize data, true, this is what we try to recover
dims     = [50, 50, 10, 10] # 10 * 10 * 10 data
hidden_D = 20
means    = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 3, np.ones((hidden_D,)) * 2, np.ones((hidden_D,)) * 8]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2, np.eye(hidden_D) * 1]
data = Tensor()
data.synthesize(dims, means, covariances, hidden_D, 0.2, 1)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20
# Likelihood function for real valued data
p_likelihood = distribution("normal", 1, ("var"), 1)

# likelihood funcion for binary valued data
# p_likelihood = distribution("bernoulli", 1, None, None)

# likelihood function for count-valued data

# Approximate posterior initial
approximate_mean = np.ones((D,)) * 5
approximate_cov = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean, approximate_cov))

# Model prior
m = np.ones((D,))
S = np.eye(D)
p_prior = distribution("normal", 1, ("approximate_mean_0", "sigma"), (m, S))
model = SSVI_TF(p_likelihood, q_posterior, p_prior)

############################### FACTORIZATION ##########################

rho_cov = lambda t: 1/(1+t)
k1 = 1
k2 = 10

factorizer = SSVI_TF(model, data, rank=D, rho_cov=rho_cov)
factorizer.factorize()
