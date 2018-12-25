import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import ModuleList, ParameterList, Parameter
import copy

from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.distributions.kl as KL

import numpy as np
import math


"""

Natural parameter update

Stores
(diagonal covariance)
S, m

Still need to compute
dT/dS and dT/dm

mean parameter update:
inv(S)m = inv(S)m + p (m - inv(S)m - dT/dS m + dT/dS)
>>> multiplying both sides with S
m  =  m + p(Sm - m - S dT/dS m + S dT/dS)

Covariance parameter update:
0.5 inv(S) = 0.5 inv(S) + p (S - 0.5 inv (S) + dT/dS)

>>> Update formula
S = inv(S + p (2S - inv(S) + 2dT/dS)

The question is 
>> how to run backward() and then modify the computed gradient value
>> Loop through the means and covs value
>> 

"""

class SSVI_torch(torch.nn.Module):
    def __init__(self, tensor, gradient_update="S", rank=10):
        super().__init__()

        self.tensor = tensor
        self.num_train = len(tensor.train_vals)
        self.dims = tensor.dims
        self.ndim = len(self.dims)
        self.rank = rank
        self.datatype = tensor.datatype
        self.gradient_update = gradient_update

        self.means = ModuleList()
        self.chols = ModuleList()

        for dim, ncol in enumerate(self.dims):
            mean_list = ParameterList()
            cov_list  = ParameterList()
            for _ in range(ncol):
                mean_list.append(Parameter(torch.randn(rank), requires_grad=True))
                cov_list.append(Parameter(torch.ones(rank) + 1/4 * torch.randn(rank), requires_grad=True))

            self.means.append(mean_list)
            self.chols.append(cov_list)

        self.standard_multi_normal = MultivariateNormal(torch.zeros(rank), torch.eye(rank))
        self.sigma = 1
        self.batch_size = 64
        self.lambd = 1/self.batch_size
        self.round_robins_indices = [0 for _ in self.dims]
        self.k1 = 128

    def factorize(self, maxiters, algorithm="Adagrad", lr=1, report=[], interval=50, round_robins=True):
        """

        :param maxiters:
        :param algorithm:
        :param lr:
        :param report:
        :param interval:
        :return:
        """
        if algorithm == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr)
        elif algorithm == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr, weight_decay=0.01)
        elif algorithm == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr, weight_decay=0.01)
        elif algorithm == "Adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr)
        elif algorithm == "Adamax":
            optimizer = torch.optim.Adamax(self.parameters(), lr)
        elif algorithm == "RMSProp":
            optimizer = torch.optim.RMSprop(self.parameters(), lr)

        with torch.enable_grad():
            start = 0
            expectation_term = 0.
            kl_term = 0.

            for iteration in range(maxiters):
                if iteration in report or iteration % interval == 0:
                    self.evaluate(iteration, expectation_term, kl_term)

                for dim, col in enumerate(self.round_robins_indices):
                    # optimizer.zero_grad()

                    # Get mini-batch in round-robbins
                    if round_robins:
                        observed_subset, num_observed_i = self.get_mini_batch_round_robins(dim, col)
                    # Get mini-batch from random order
                    else:
                        observed_subset = self.get_mini_batch(start)
                        start = (start + self.batch_size) % len(self.tensor.train_vals)

                    expectation_term, kl_term = self.elbo_compute(observed_subset)
                    num_sample = len(observed_subset)

                    # Natural gradient
                    if self.gradient_update == "N":
                        # self.natural_gradient_update(observed_subset, expectation_term, kl_term, optimizer)
                        # self.natural_gradient_update_round_robins(expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col)
                        # self.natural_gradient_update_v2(observed_subset, expectation_term, kl_term, optimizer)
                        # self.natural_gradient_update_round_robins_v2(expectation_term, kl_term, optimizer, num_sample,num_observed_i, dim, col)
                        self.natural_gradient_update_round_robins_mean_only(expectation_term, kl_term, optimizer, num_sample,num_observed_i, dim, col)

                    # Hybrid update
                    elif self.gradient_update == "H":
                        # self.hybrid_gradient_update(observed_subset, expectation_term, kl_term, optimizer)
                        # self.hybrid_gradient_update_v2(observed_subset, expectation_term, kl_term, optimizer)
                        self.hybrid_gradient_update_round_robins(expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col)

                    # Standard gradient update
                    else:
                        # loss = -self.num_train/self.batch_size * expectation_term + kl_term
                        # loss = kl_term - expectation_term
                        # loss.backward()
                        # optimizer.step()
                        # optimizer.zero_grad()
                        self.standard_gradient_update_round_robins(expectation_term, kl_term, optimizer, dim, col)
                        # self.standard_gradient_sanity_check(observed_subset, expectation_term, kl_term, optimizer)

                for dim, col in enumerate(self.round_robins_indices):
                    self.round_robins_indices[dim] += 1
                    self.round_robins_indices[dim] %= self.dims[dim]

    def get_mini_batch(self, start):
        end = (start + self.batch_size) % len(self.tensor.train_vals)
        if end > start:
            observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, end)]
        else:
            observed_subset = [(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(start, len(self.tensor.train_vals))]
            observed_subset.extend([(self.tensor.train_entries[i], self.tensor.train_vals[i]) for i in range(end)])
        return observed_subset

    def get_mini_batch_round_robins(self, dim, col):
        observed = self.tensor.find_observed_ui(dim, col)
        if len(observed) > self.batch_size:
            observed_idx = np.random.choice(len(observed), self.batch_size, replace=False)
            observed_subset = np.take(observed, observed_idx, axis=0)
        else:
            observed_subset = observed
        return observed_subset, len(observed)

    def standard_gradient_update_round_robins(self, expectation_term, kl_term, optimizer, dim, col):
        # loss = -self.num_train/self.batch_size * expectation_term + kl_term
        loss = -expectation_term + kl_term
        loss.backward()

        dm = copy.deepcopy(self.means[dim][col].grad)
        dL = copy.deepcopy(self.chols[dim][col].grad)

        # TODO: note that if I set dm, dL the way above, it does gets updated with the zero_grad!!!
        optimizer.zero_grad() # Remove the grad of other factors
        if not torch.any(torch.isnan(dm)):
            self.means[dim][col].grad = dm
        if not torch.any(torch.isnan(dL)):
            self.chols[dim][col].grad = dL

        optimizer.step()

        # Check if any entries in L are negative
        L_new = copy.deepcopy(self.chols[dim][col].data)
        L_pos_mean = torch.mean(L_new[L_new > 0])
        L_new[L_new <= 0] = L_pos_mean
        self.chols[dim][col].data = L_new


    def natural_gradient_update_round_robins(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        """
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:
        """
        # Automatically compute the derivative with respect to the parameters
        expectation_term.backward()

        dm = copy.deepcopy(self.means[dim][col].grad)
        dL = copy.deepcopy(self.chols[dim][col].grad)
        m = copy.deepcopy(self.means[dim][col].data)
        L = copy.deepcopy(self.chols[dim][col].data)

        # Compute the natural gradient required for natural parameter
        # Note that pytorch does MINUS gradient * stepsize
        # and we're doing gradient ascent
        # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1
        # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
        scale = num_observed_i/num_sample

        G_mean = (dm - dL * m / L)
        natural_mean_grad = (m - m / L ** 2 - scale * G_mean) * -1

        G_chol = -dL * 1 / (2 * L)
        natural_chol_grad = ((L ** 2) - 0.5 * 1 / (L ** 2) + scale * G_chol) * -1

        optimizer.zero_grad() # Remove the grad of other factors

        # Replace the gradient with the natural gradient
        nat_update_for_mean = False
        if not torch.any(torch.isnan(natural_mean_grad)):
            self.means[dim][col].grad = natural_mean_grad
            nat_update_for_mean = True
        else:
            self.means[dim][col].grad = -dm


        # If "unstable" do a standard gradient update
        nat_update_for_chol = False
        if not torch.any(torch.isnan(natural_chol_grad)) and nat_update_for_mean:
            self.chols[dim][col].grad = natural_chol_grad
            nat_update_for_chol = True
        else:
            self.chols[dim][col].grad = -dL

        # Compute the natural parameters for the affected columns
        if nat_update_for_mean:
            # m -> m/L**2
            self.means[dim][col].data = m/(L**2)

        if nat_update_for_chol:
            # L -> 0.5 /L**2
            self.chols[dim][col].data = 0.5 / (L**2)

        optimizer.step()

        if nat_update_for_chol and nat_update_for_mean:
            # The current covariance parameter being stored is 0.5/L**2
            # 0.5/L**2 = x => L = sqrt(0.5/x)
            L_natural = copy.deepcopy(self.chols[dim][col].data)
            L_squared = 0.5 / L_natural
            # L_squared = F.relu(L_squared) + 1e-4

            # The current mean parameter being stored is m/L**2
            # m/L**2 =   => m = L**2 x
            m_natural = copy.deepcopy(self.means[dim][col].data)
            m_new = L_squared * m_natural
            L_new = torch.sqrt(L_squared)

            # Convert from natural parameter form to standard form
            if not torch.any(torch.isnan(m_new)) and not torch.any(torch.isinf(m_new)):
                self.means[dim][col].data = m_new
            else:
                self.means[dim][col].data = m

            if not torch.any(torch.isnan(L_new)):
                self.chols[dim][col].data = L_new
            else:
                self.chols[dim][col].data = L


    def natural_gradient_update_round_robins_v2(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        """
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:
        """
        # Automatically compute the derivative with respect to the parameters
        expectation_term.backward(retain_graph=True)

        dm = copy.deepcopy(self.means[dim][col].grad)
        dL = copy.deepcopy(self.chols[dim][col].grad)
        m = copy.deepcopy(self.means[dim][col].data)
        L = copy.deepcopy(self.chols[dim][col].data)

        optimizer.zero_grad()

        # Compute the natural gradient required for natural parameter
        scale = num_observed_i / num_sample

        # Compute the natural gradient
        G_mean = (dm - dL * m / L) # why is this -dL m /L
        natural_mean_grad = (- m / (L ** 2) + m + scale * G_mean) * -1

        G_chol = -dL * 1 / (2 * L)
        natural_chol_grad = (- 0.5 * 1 / (L ** 2) + (L ** 2) + scale * G_chol) * -1

        # Note that pytorch does MINUS gradient * stepsize
        # and we're doing gradient ascent
        # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1
        # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
        if torch.any(torch.isnan(natural_mean_grad)) or torch.any(torch.isnan(natural_chol_grad)):
            print("standard!")
            self.standard_gradient_update_round_robins(expectation_term, kl_term, optimizer, dim, col)

        else:
            # Replace the gradient with the natural gradient
            self.means[dim][col].grad = natural_mean_grad
            self.chols[dim][col].grad = natural_chol_grad

            # Compute the natural parameters for the affected columns
            self.means[dim][col].data = m/(L**2)
            self.chols[dim][col].data = 0.5 / (L**2)

            optimizer.step()

            # Compute the standard parameters from the current natural parameters
            L_natural = copy.deepcopy(self.chols[dim][col].data)
            L_squared = 0.5 / L_natural

            mean_L_squared_pos = torch.mean(L_squared[L_squared > 0])
            L_squared[L_squared < 0 ] = mean_L_squared_pos

            # The current mean parameter being stored is m/L**2
            # m/L**2 =   => m = L**2 x
            m_natural = copy.deepcopy(self.means[dim][col].data)
            m_new = L_squared * m_natural
            L_new = torch.sqrt(L_squared)

            # Convert from natural parameter form to standard form
            self.means[dim][col].data = m_new

            # Check that L_new doesn't have any negative entries
            L_pos_mean = torch.mean(L_new[L_new > 0])
            L_new[L_new <= 0] = L_pos_mean
            L_new = torch.max(L_new, torch.FloatTensor([1]))
            self.chols[dim][col].data = L_new


    def natural_gradient_update_round_robins_mean_only(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        """
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:
        """
        # Automatically compute the derivative with respect to the parameters
        expectation_term.backward(retain_graph=True)

        dm = copy.deepcopy(self.means[dim][col].grad)
        dL = copy.deepcopy(self.chols[dim][col].grad)
        m = copy.deepcopy(self.means[dim][col].data)
        L = copy.deepcopy(self.chols[dim][col].data)

        optimizer.zero_grad()

        # Compute the natural gradient required for natural parameter
        scale = num_observed_i / num_sample
        scale = 1

        # Compute the natural gradient
        G_mean = (dm + dL * m / L) # why is this -dL m /L
        natural_mean_grad = (- m / (L ** 2) + m + scale * G_mean) * -1
        G_chol = -dL * 1 / (2 * L)
        natural_chol_grad = (- 0.5 * 1 / (L ** 2) + (L ** 2) + scale * G_chol) * -1

        # Note that pytorch does MINUS gradient * stepsize
        # and we're doing gradient ascent
        # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1
        # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm

        # Replace the gradient with the natural gradient
        self.means[dim][col].grad = natural_mean_grad
        # self.means[dim][col].grad = -dm
        # Compute the NATURAL parameters for the affected columns

        self.means[dim][col].data = m/(L**2)

        optimizer.step()

        m_natural = copy.deepcopy(self.means[dim][col].data)
        m_new = (L**2) * m_natural
        # Convert from natural parameter form to standard form
        self.means[dim][col].data = m_new

    def natural_gradient_update(self, observed_subset, expectation_term, kl_term, optimizer):
        """
        :param observed_subset:
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:

        Mean update
        1/(L**2)m  = 1/(L**2)m + p [ m - 1/(L**2) + dT/dm + dT/dL m/L ]
        0.5 / L**2 = 0.5 / L**2 + p [ L**2 - 1/(2L**2) - dT/dL 1/(2L) ]

        TODO: scaling problem

        """
        entries = [pair[0] for pair in observed_subset]
        expectation_term.backward()

        L_prev = list()

        rows_by_dim = list()
        for dim, _ in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            rows_by_dim.append(all_rows)

        for dim, _ in enumerate(self.dims):
            all_rows = rows_by_dim[dim]

            # Get the GRADIENT of the expectation term wrt
            # the MEAN and CHOLESKY parameters

            dm = copy.deepcopy(torch.stack([self.means[dim][row].grad for row in all_rows], dim=0))
            dL = copy.deepcopy(torch.stack([self.chols[dim][row].grad for row in all_rows], dim=0))
            m  = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            L  = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            # dm = copy.deepcopy(self.means._parameters[str(dim)].grad[all_rows, :])
            # dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_rows, :])
            # m  = copy.deepcopy(self.means._parameters[str(dim)].data[all_rows, :])
            # L  = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])

            # Compute the natural gradient required for natural parameter
            # Note that pytorch does MINUS gradient * stepsize
            # and we're doing gradient ascent
            # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1
            # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
            # scale = self.batch_size
            scale = 1

            G_mean =  dm - dL * m / L
            natural_mean_grad = (m - m / L ** 2 + scale * G_mean) #* -1

            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/L**2 + scale * G_chol) #* -1

            # Precompute the current natural parameters
            # m -> m/L**2
            # self.means._parameters[str(dim)].data[all_rows, :] = m/L**2
            # # L -> 0.5 /L**2
            # self.chols._parameters[str(dim)].data[all_rows, :] = 0.5 * L**2
            # # Replace the gradient with the natural gradient
            # self.means._parameters[str(dim)].grad[all_rows, :] = natural_mean_grad
            # self.chols._parameters[str(dim)].grad[all_rows, :] = natural_chol_grad

            for i, row in enumerate(all_rows):
                self.means[dim][row].data = m[i, :] / L[i, :] ** 2
                # L -> 0.5 /L**2
                self.chols[dim][row].data = 0.5 * L[i, :] ** 2

                # Replace the gradient with the natural gradient
                self.means[dim][row].grad = natural_mean_grad[i, :]
                self.chols[dim][row].grad = natural_chol_grad[i, :]

        # Do one step of update
        optimizer.step()

        # Re-update the parameters
        for dim, ncol in enumerate(self.dims):
            all_rows = rows_by_dim[dim]

            # The current covariance parameter being stored is 0.5/L**2
            # 0.5/L**2 = x => L = sqrt(0.5/x)
            # L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])
            L_natural = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            L_squared = 0.5/L_natural
            # L_squared = F.relu(L_squared) + 1e-4
            L_squared = torch.max(L_squared, torch.FloatTensor([1e-4]))

            # The current mean parameter being stored is m/L**2
            # m/L**2 =   => m = L**2 x
            # m_natural = copy.deepcopy(self.means._parameters[str(dim)].data[all_rows, :])
            m_natural = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            m_new = L_squared * m_natural

            L_new = torch.sqrt(L_squared)
            # L[torch.isnan(L)] = 0.1
            # self.means._parameters[str(dim)].data[all_rows, :] = m_new
            # self.chols._parameters[str(dim)].data[all_rows, :] = L_new

            for i, row in enumerate(all_rows):
                self.means[dim][row].data = m_new[i, :]
                self.chols[dim][row].data = L_new[i, :]

    def natural_gradient_update_v2(self, observed_subset, expectation_term, kl_term, optimizer):
        """
        :param observed_subset:
        :param expectation_term:
        :param kl_term:
        :param optimizer:
        :param num_sample:
        :param num_observed_i:
        :param dim:
        :param col:
        :return:

        Mean update
        1/(L**2)m  = 1/(L**2)m + p [ m - 1/(L**2) + dT/dm + dT/dL m/L ]
        0.5 / L**2 = 0.5 / L**2 + p [ L**2 - 1/(2L**2) - dT/dL 1/(2L) ]

        """
        entries = [pair[0] for pair in observed_subset]
        scaled_expectation = self.num_train/self.batch_size * expectation_term
        # expectation_term.backward()
        scaled_expectation.backward()

        L_prev = list()

        rows_by_dim = list()
        for dim, _ in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            rows_by_dim.append(all_rows)

        dm_list = []
        dL_list = []
        m_list  = []
        L_list  = []

        for dim, _ in enumerate(self.dims):
            all_rows = rows_by_dim[dim]

            # Get the GRADIENT of the expectation term wrt
            # the MEAN and CHOLESKY parameters
            dm = copy.deepcopy(torch.stack([self.means[dim][row].grad for row in all_rows], dim=0))
            dL = copy.deepcopy(torch.stack([self.chols[dim][row].grad for row in all_rows], dim=0))
            m  = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            L  = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            scale = 1

            G_mean =  dm - dL * m / L
            natural_mean_grad = (m - m / (L ** 2) + scale * G_mean) * -1

            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/(L**2) + scale * G_chol) * -1

            dm_list.append(natural_mean_grad)
            dL_list.append(natural_chol_grad)
            m_list.append(m)
            L_list.append(L)

        # Remove the gradient
        optimizer.zero_grad()

        for dim, _ in enumerate(self.dims):
            # Precompute the current natural parameters
            # m -> m/L**2
            # self.means._parameters[str(dim)].data[all_rows, :] = m/L**2
            # # L -> 0.5 /L**2
            all_rows = rows_by_dim[dim]
            m = m_list[dim]
            L = L_list[dim]
            natural_mean_grad = dm_list[dim]
            natural_chol_grad = dL_list[dim]

            for i, row in enumerate(all_rows):
                self.means[dim][row].data = m[i, :] / (L[i, :] ** 2)
                # L -> 0.5 /L**2
                self.chols[dim][row].data = 0.5 * (L[i, :] ** 2)
                # Replace the gradient with the natural gradient
                self.means[dim][row].grad = natural_mean_grad[i, :]
                self.chols[dim][row].grad = natural_chol_grad[i, :]

        # Do one step of update
        optimizer.step()

        # Re-update the parameters
        for dim, ncol in enumerate(self.dims):
            all_rows = rows_by_dim[dim]

            # The current covariance parameter being stored is 0.5/L**2
            # 0.5/L**2 = x => L = sqrt(0.5/x)
            L_natural = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            L_squared = 0.5/L_natural
            L_squared = F.relu(L_squared) + 1e-4
            # L_squared = torch.max(L_squared, torch.FloatTensor([1e-4]))

            # The current mean parameter being stored is m/L**2
            # m/L**2 =   => m = L**2 x
            m_natural = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            m_new = L_squared * m_natural

            L_new = torch.sqrt(L_squared)
            # L[torch.isnan(L)] = 0.1

            for i, row in enumerate(all_rows):
                self.means[dim][row].data = m_new[i, :]
                self.chols[dim][row].data = L_new[i, :]

    def hybrid_gradient_update(self, observed_subset, expectation_term, kl_term, optimizer):
        # If using natural gradient
        entries = [pair[0] for pair in observed_subset]

        # Batch loss
        # batch_loss = - self.num_train/self.batch_size * expectation_term + kl_term

        batch_loss = -expectation_term
        batch_loss.backward(retain_graph=True)

        mean_grad = list()

        # Zeros out the Cholesky factor
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            # Get the GRADIENT of total loss with respect to the CHOLESKY factors
            # dm = copy.deepcopy(self.means._parameters[str(dim)].grad[all_cols, :])
            dm = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            mean_grad.append(dm)

        # Update the cholesky factor via natural gradient
        # Loss term from expectation
        optimizer.zero_grad()

        expectation_term.backward()
        # Update the Cholesky factors via Natural gradient
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)

            # Get the GRADIENT of the expectation term wrt
            # CHOLESKY parameters
            # dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_rows, :])
            # L  = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])
            dL = copy.deepcopy(torch.stack([self.chols[dim][row].grad for row in all_rows], dim=0))
            L = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            # Replace the gradient for the mean with previously computed value
            # self.means._parameters[str(dim)].grad[all_rows, :] = mean_grad[dim]
            for i, row in enumerate(all_rows):
                self.means[dim][row].data = mean_grad[dim][i, :]

            # Compute the natural gradient required for natural parameter
            scale = 1
            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/L**2 + scale * G_chol) * -1

            # Replace the gradient with the natural gradient
            # self.chols._parameters[str(dim)].grad[all_rows, :] = natural_chol_grad

            for i, row in enumerate(all_rows):
                self.chols[dim][row].grad = natural_chol_grad[i,:]
                self.chols[dim][row].data = 1/L[i,:]**2
            # Compute the current natural parameters
            # self.chols._parameters[str(dim)].data[all_rows, :] = 1/L**2

        # Do one step of update
        optimizer.step()

        # Re-Flip the Cholesky decomposition
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            # L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])
            L_natural = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))
            L_squared = 0.5/L_natural
            L_new     = torch.sqrt(F.relu(L_squared)) + 1e-4

            for i, row in enumerate(all_rows):
                self.chols[dim][row].data = L_new[i,:]
            # self.chols._parameters[str(dim)].data[all_rows, :] = L_new

    def hybrid_gradient_update_v2(self, observed_subset, expectation_term, kl_term, optimizer):
        # If using natural gradient
        entries = [pair[0] for pair in observed_subset]

        # Batch loss
        batch_loss = - self.num_train/self.batch_size * expectation_term + kl_term
        batch_loss.backward(retain_graph=True)

        mean_grad = list()

        # Zeros out the Cholesky factor
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            # Get the GRADIENT of total loss with respect to the CHOLESKY factors
            # dm = copy.deepcopy(self.means._parameters[str(dim)].grad[all_cols, :])
            dm = copy.deepcopy(torch.stack([self.means[dim][row].data for row in all_rows], dim=0))
            mean_grad.append(dm)

        # Update the cholesky factor via natural gradient
        # Loss term from expectation
        optimizer.zero_grad()

        expectation_term.backward()
        # Update the Cholesky factors via Natural gradient
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)

            # Get the GRADIENT of the expectation term wrt
            # CHOLESKY parameters
            # dL = copy.deepcopy(self.chols._parameters[str(dim)].grad[all_rows, :])
            # L  = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])
            dL = copy.deepcopy(torch.stack([self.chols[dim][row].grad for row in all_rows], dim=0))
            L = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))

            # Replace the gradient for the mean with previously computed value
            # self.means._parameters[str(dim)].grad[all_rows, :] = mean_grad[dim]
            for i, row in enumerate(all_rows):
                self.means[dim][row].data = mean_grad[dim][i, :]

            # Compute the natural gradient required for natural parameter
            scale = 1
            G_chol = -dL * 1/ (2*L)
            natural_chol_grad = (L**2 - 0.5 * 1/L**2 + scale * G_chol) * -1

            # Replace the gradient with the natural gradient
            # self.chols._parameters[str(dim)].grad[all_rows, :] = natural_chol_grad

            for i, row in enumerate(all_rows):
                self.chols[dim][row].grad = natural_chol_grad[i,:]
                self.chols[dim][row].data = 1/L[i,:]**2
            # Compute the current natural parameters
            # self.chols._parameters[str(dim)].data[all_rows, :] = 1/L**2

        # Do one step of update
        optimizer.step()

        # Re-Flip the Cholesky decomposition
        for dim, ncol in enumerate(self.dims):
            all_rows = set([x[dim] for x in entries])
            all_rows = list(all_rows)
            # L_natural = copy.deepcopy(self.chols._parameters[str(dim)].data[all_rows, :])
            L_natural = copy.deepcopy(torch.stack([self.chols[dim][row].data for row in all_rows], dim=0))
            L_squared = 0.5/L_natural
            L_new     = torch.sqrt(F.relu(L_squared)) + 1e-4

            for i, row in enumerate(all_rows):
                self.chols[dim][row].data = L_new[i,:]
            # self.chols._parameters[str(dim)].data[all_rows, :] = L_new

    def hybrid_gradient_update_round_robins(self, expectation_term, kl_term, optimizer, num_sample, num_observed_i, dim, col):
        # Automatically compute the derivative with respect to the parameters
        # standard_loss = -self.num_train/self.batch_size * expectation_term + kl_term
        standard_loss = -expectation_term + kl_term
        standard_loss.backward(retain_graph=True)

        dm = copy.deepcopy(self.means[dim][col].grad)

        # Compute the natural gradient required for natural parameter
        # Note that pytorch does MINUS gradient * stepsize
        # and we're doing gradient ascent
        # natural_mean_grad = (m - m / L**2 + dm + dL * m/L) * -1
        # Roni's derivations: dT/dh = dT/dm - 2dT/dS dm
        # Remove the gradients computed using standard loss
        optimizer.zero_grad()

        # Implicitly compute the gradient with respect to the "NATURAL" cholesky factor
        scaled_expectation = self.num_train/self.batch_size * expectation_term
        scaled_expectation.backward()
        dL = copy.deepcopy(self.chols[dim][col].grad)
        L = copy.deepcopy(self.chols[dim][col].data)

        # Compute the natural gradient
        scale = num_observed_i/num_sample
        G_chol = -dL * 1 / (2 * L)
        natural_chol_grad = ((L ** 2) - 0.5 * 1 / (L ** 2) + scale * G_chol) * -1

        # Precompute the natural parameters for thhe cholesky factor
        # L -> 0.5 /L**2
        # self.chols._parameters[str(dim)].data[col, :] = 0.5 / L ** 2
        self.chols[dim][col].data = 0.5 / (L**2)

        optimizer.zero_grad()
        # Replace the gradient with the natural gradient
        self.means[dim][col].grad = dm
        if not torch.any(torch.isnan(natural_chol_grad)):
            self.chols[dim][col].grad = natural_chol_grad

        optimizer.step()

        # The current covariance parameter being stored is 0.5/L**2
        # 0.5/L**2 = x => L = sqrt(0.5/x)
        L_natural = copy.deepcopy(self.chols[dim][col].data)
        L_squared = 0.5 / L_natural
        L_squared = F.relu(L_squared) + 1e-4

        # The current mean parameter being stored is m/L**2
        # m/L**2 =   => m = L**2 x
        L_new = torch.sqrt(L_squared)
        # If nan -> just use the previous L
        if not torch.any(torch.isnan(L_new)):
            self.chols[dim][col].data = L_new
        else:
            self.chols[dim][col].data = L

    def elbo_compute(self, observed):
        """
        :param observed:
        :return:
        """
        entries = [pair[0] for pair in observed]
        ys = Variable(torch.FloatTensor([pair[1] for pair in observed]))

        # entries = list of coordinates
        # y = vector of entry values
        # Compute the expectation as a batch
        batch_expectation = self.compute_batch_expectation_term(entries, ys)

        # Compute the KL term as a batch
        batch_kl = self.compute_batch_kl_term(entries)

        # loss -= self.num_train/self.batch_size * batch_expectation + (1/ (self.batch_size * self.ndim)) * batch_kl
        expectation_term = batch_expectation
        kl_loss          =  (1 / (self.batch_size )) * batch_kl

        return expectation_term, kl_loss

    def compute_batch_expectation_term(self, entries, ys):
        num_samples = len(entries)
        ndim = len(self.dims)

        element_mult_samples = torch.ones(num_samples, self.k1, self.rank) # shape = (num_samples, k1, rank)
        for dim, nrow in enumerate(self.dims):
            all_rows = [x[dim] for x in entries]
            # stack these rows into a matrix
            all_ms = torch.stack([self.means[dim][row] for row in all_rows], dim=0)
            all_Ls = torch.stack([self.chols[dim][row] for row in all_rows], dim=0)

            sampler = MultivariateNormal(torch.zeros(self.rank), torch.eye(self.rank))
            epsilon_tensor = sampler.sample((num_samples, self.k1))

            # Create k1 copies (rows of all_ms)
            all_ms.unsqueeze_(-1)
            ms_copied = all_ms.expand(num_samples, self.rank, self.k1)
            ms_copied = ms_copied.transpose(2, 1)

            for num in range(num_samples):
                L_squared = all_Ls[num, :]**2 # shape = (rank)
                eps_term  =  epsilon_tensor[num, :, :] # shape = (k1, rank)
                var_term  = eps_term * L_squared # shape = (k1, rank)
                element_mult_samples[num, :, :] *= ms_copied[num, :, :] + var_term

        # fs_samples.shape = (num_samples, k1)
        fs_samples = element_mult_samples.sum(dim=2) # sum along the 3rd dimension (along rank)

        target_vector = Variable(torch.FloatTensor(ys))
        target_vector = target_vector.view(num_samples, 1)
        target_matrix = target_vector.repeat(1, self.k1)

        # Compute log pdf
        log_pdf = self.compute_log_pdf(fs_samples, target_matrix)

        expected_log_pdf = log_pdf.mean(dim=1)

        batch_expectation = expected_log_pdf.sum()
        return batch_expectation

    def compute_batch_kl_term(self, entries):
        kl = 0.

        for dim, _ in enumerate(self.dims):
            all_rows = [x[dim] for x in entries]
            # all_cols = Variable(torch.LongTensor(all_cols))

            all_ms = torch.stack([self.means[dim][row] for row in all_rows], dim=0)
            all_Ls = torch.stack([self.chols[dim][row] for row in all_rows], dim=0)
            all_S = all_Ls ** 2

            # KL(N0 || N1) = 0.5 (tr(S1^-1 S0) + (m1-m0)^T S1^-1 (m1-m0) - D + ln(|S_1|/|S_0|)
            # When N1 = N(0, I)
            # KL(N0 || N(0, I)) = 0.5 (tr(S0) + m0^T m0 - D + ln(1/|S_0|)

            kl_batch = 0.5 * torch.sum(-self.rank + torch.sum(all_S, dim=1) + torch.sum(all_ms**2, dim=1) - torch.log(1/torch.prod(all_S, dim=1)))

            # all_S  = all_Ls ** 2
            # kl_div = KL._kl_normal_normal(Normal(all_ms, all_S), Normal(0, 1))
            # kl_div = torch.sum(kl_div)
            # kl -= kl_div
            # kl -= 0.5 * torch.sum(1 + torch.log(all_S ** 2) - all_ms**2 - all_S**2)

        return kl

    def compute_log_pdf(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        # fs_samples.shape = (num_samples, k1)
        # target_matrix.shape = (num_samples, k1)
        if self.datatype == "real":
            return self.compute_log_pdf_normal(fs_samples, target_matrix)

        elif self.datatype == "binary":
            return self.compute_log_pdf_bernoulli(fs_samples, target_matrix)

        elif self.datatype == "count":
            return self.compute_log_pdf_poisson(fs_samples, target_matrix)

        return log_pdf

    def compute_log_pdf_normal(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        # dist = Normal(fs_samples, 1)
        # log_pdf = dist.log_prob(target_matrix)
        log_pdf = (-0.5 * (fs_samples - target_matrix)**2)

        return log_pdf

    def compute_log_pdf_bernoulli(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:
        """
        dist = Bernoulli(torch.sigmoid(fs_samples))
        log_pdf = dist.log_prob(target_matrix)
        return log_pdf

    def compute_log_pdf_poisson(self, fs_samples, target_matrix):
        """

        :param fs_samples:
        :param target_matrix:
        :return:

        This gives the most trouble because of
        """
        lamb = F.relu(fs_samples, inplace=True)
        # Compute log likelihood
        dist = Poisson(lamb)
        log_pdf = dist.log_prob(target_matrix)

        nan_indices = torch.isnan(log_pdf)
        log_pdf[nan_indices] = -100
        inf_indices = torch.isinf(log_pdf)
        log_pdf[inf_indices] = -100
        neg_inf_indices = torch.isinf(-log_pdf)
        log_pdf[neg_inf_indices] = -100

        return log_pdf

    def evaluate(self, iteration, expectation, kl):
        if iteration == 0:
            print(" iteration | test mae  | train mae |  E-term  |    KL   |")

        train_mae = self.evaluate_train_error()
        test_mae = self.evaluate_test_error()
        # expectation_term = expectation.detach().numpy()
        # kl_term = kl.detach().numpy()
        print ("{:^10} {:^10} {:^10} {:^10} {:^10}".format(iteration, np.around(test_mae, 4), np.around(train_mae, 4), \
                                                    expectation, kl))

    def evaluate_train_error(self):
        if self.datatype == "binary":
            return self.evaluate_error_rate(self.tensor.train_entries, self.tensor.train_vals)
        return self.evaluate_mae(self.tensor.train_entries, self.tensor.train_vals)

    def evaluate_test_error(self):
        if self.datatype == "binary":
            return self.evaluate_error_rate(self.tensor.test_entries, self.tensor.test_vals)
        return self.evaluate_mae(self.tensor.test_entries, self.tensor.test_vals)

    def evaluate_mae(self, entries, vals):
        """
        :param entries:
        :param vals:
        :return:
        """
        mae = 0.0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            predict = self.predict_entry(entry)
            correct = vals[i]
            mae += abs(predict - correct)

        mae = mae/num_entries
        return mae

    def evaluate_error_rate(self, entries, vals):
        """
        :param entries:
        :param vals:
        :return:
        """
        error = 0
        num_entries = len(vals)

        for i in range(len(entries)):
            entry = entries[i]
            predict = self.predict_entry(entry)
            correct = vals[i]
            if correct != predict:
                error += 1

        return float(error)/num_entries

    def predict_entry(self, entry):
        # TODO: generalize to other likelihood types
        inner = torch.ones(self.rank)
        for dim, col in enumerate(entry):
            # inner *= self.means[dim][col, :]
            inner *= self.means[dim][col]

        if self.datatype == "real":
            return float(torch.sum(inner))
        elif self.datatype == "binary":
            return 1 if torch.sum(inner) > 0 else -1
        elif self.datatype == "count":
            return float(torch.sum(F.relu(inner)))