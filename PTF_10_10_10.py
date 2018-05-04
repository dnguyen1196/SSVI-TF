
import numpy as np
# np.random.seed(123)

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF import H_SSVI_TF_2d
from Tensor.Tensor import tensor


# Generate synthesize tensor, true, this is what we try to recover
dims     = [10, 10, 10] # 10 * 10 * 10 tensor
hidden_D = 20
means    = [np.ones((hidden_D,)) * 5, np.ones((hidden_D,)) * 3, np.ones((hidden_D,)) * 2]
covariances = [np.eye(hidden_D) *2, np.eye(hidden_D) * 3, np.eye(hidden_D) * 2]
data = tensor()
data.synthesize(dims, means, covariances, hidden_D, 0.8, 0.50)


################## MODEL and FACTORIZATION #########################
# The below two modules have to agree with one another on dimension
# Generate a simple TF_model

D = 20
# Likelihood function
p_likelihood = distribution("normal", 1, ("var"), 1)

# Approximate posterior initial
approximate_mean = np.ones((D,))
approximate_cov = np.eye(D)
q_posterior = distribution("normal", dims, ("mean", "cov"), (approximate_mean, approximate_cov))

# Model prior
m = np.ones((D,))
S = np.eye(D)
p_prior = distribution("normal", 1, ("approximate_mean", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior)

############################### FACTORIZATION ##########################

rho_mean = lambda t : 1/(t + 1)
rho_cov  = lambda t : 0.05
k1 = 25
k2 = 10

factorizer = H_SSVI_TF_2d(model, data, rank=D, rho_mean=rho_mean, rho_cov=rho_cov, k1=k1, k2=k2)
factorizer.factorize()
# #
# # print("Approximate: ", model.q_posterior.find(0, 0)[0])
# print("Actual: ", data.matrices[0][0, :])
# print("MSRE: ", factorizer.evaluate())