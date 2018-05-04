import numpy as np
# np.random.seed(123)

from Model.TF_Models import SSVI_TF_d, distribution
from SSVI.SSVI_TF import H_SSVI_TF_2d
from Tensor.Tensor import tensor


# Generate synthesize tensor, true, this is what we try to recover
dims        = [20, 20] # 20 * 20 matrix
hidden_D    = 10
means = [np.ones((hidden_D,)), np.ones((hidden_D,))]
covariances = [np.eye(hidden_D), np.eye(hidden_D)]
data = tensor()
data.synthesize(dims, means, covariances, hidden_D, 0.8, 0.5)


# The below two modules have to agree with one another on dimension
# Generate a simple TF_model
D = 10
p_likelihood = distribution("normal", 1, ("var"), 1)
mu = np.ones((D,)) * 1.1
Su = np.eye(D) * 1.1
q_posterior = distribution("normal", dims, ("mean", "cov"), (mu, Su))

m = np.ones((D,)) * 1.1
S = np.eye(D) * 1.1
p_prior = distribution("normal", 1, ("approximate_mean", "sigma"), (m, S))
model = SSVI_TF_d(p_likelihood, q_posterior, p_prior)

# SSVI_TF
rho = lambda t : 1/(t + 15)
rho_const = lambda t : 0.001

rho_mean = lambda t : 1/(10*t)
rho_cov  = lambda t : 0.1

k1 = 25
k2 = 10

factorizer = H_SSVI_TF_2d(model, data, rank=10, rho_mean=rho_mean, rho_cov=rho_cov, k1=k1, k2=k2)
factorizer.factorize()


print("Approximate: ", model.q_posterior.find(0, 0)[0])
print("Actual: ", data.matrices[0][0, :])
print("MSRE: ", factorizer.evaluate())