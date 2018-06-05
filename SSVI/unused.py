def update_natural_params(self, dim, i, iteration):
    """
    :param i:
    :param dim:
    :return:
    Update the natural parameter for the hidden column vector
    of dimension dim and column i
    """
    observed_i = self.tensor.find_observed_ui(dim, i)
    if len(observed_i) > self.batch_size:
        observed_idx = np.random.choice(len(observed_i), self.batch_size, replace=False)
        observed_i = np.take(observed_i, observed_idx, axis=0)

    (m, S) = self.model.q_posterior.find(dim, i)

    Di_acc = np.zeros((self.D, self.D))
    di_acc = np.zeros((self.D,))

    for entry in observed_i:
        coord = entry[0]
        y = entry[1]

        if self.likelihood_type == "normal":
            (di_acc_update, Di_acc_update) = self.estimate_di_Di_complete_conditional(dim, i, coord, y, m, S)
        else:
            (di_acc_update, Di_acc_update) = self.estimate_di_Di(dim, i, coord, y, m, S)

        Di_acc += Di_acc_update
        di_acc += di_acc_update

    Di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))
    di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))

    # Update covariance parameter
    covGrad = (1. / self.pSigma[dim] * np.eye(self.D) - 2 * Di_acc)
    covStep = self.compute_stepsize_cov_param(dim, i, covGrad)

    S = inv(np.multiply(np.subtract(np.ones_like(covStep), covStep), inv(S)) + np.multiply(covStep, covGrad))

    # Update mean parameter
    meanGrad = (np.inner(1. / self.pSigma[dim] * np.eye(self.D), self.pmu - m) + di_acc)
    meanStep = self.compute_stepsize_mean_param(dim, i, meanGrad)
    m += np.multiply(meanStep, meanGrad)
    self.model.q_posterior.update(dim, i, (m, S))