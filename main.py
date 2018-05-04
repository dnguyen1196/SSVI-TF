import numpy as np

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF import H_SSVI_TF_2d
from Tensor.Tensor import tensor


# Generate synthesize tensor
dims = [5, 5, 5] # 10 * 10 * 10 tensor
D    = 5
means = [np.ones((D,))*2, np.ones((D,))*3, np.ones((D,))*4]
covariances = [np.eye(D) * 2, np.eye(D) * 1, np.eye(D) * 0.5]
data = tensor()
data.synthesize(dims, means, covariances, D, 0.8, 0.25)


# The below two modules have to agree with one another on dimension
# Generate a simple TF_model
p_likelihood = distribution("normal", 1, ("var"), (5))
mu = np.ones((D,))
Su = np.eye(D)
q_posterior = distribution("normal", (5, 5, 5), ("mean", "cov"), (mu, Su))

m = np.ones((D,)) * 5
S = np.eye(D) * 5
p_prior = distribution("normal", 1, ("approximate_mean", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior)

# SSVI_TF
factorizer = H_SSVI_TF_2d(model, data, rank=D)
factorizer.factorize()