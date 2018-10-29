"""
Implement Online EM algorithm for binary and count data

The idea is that
Y ~ f(phi)
where phi is an imaginary tensor with the same size as Y

Say we can also index the observed entries Y_1, Y_2, Y_3, ... and likewise P_1, P_2, ...

The factorization model is more complicated than MCMC-SSVI since we also learn the lambda vector, call it L
Let the rank be R, and L, ui, vj, tk,... are also vectors of length R

P_i = sum(L * ui * vj * ... )

The likelihood function can be written as
f = exp(P_i)^yi / (1 + exp(P_i))^mi

where yi is the observed entry value, which is 0/1 for binary data and 0, 1, 2,... for count data
mi = yi + eps where eps is the overdispersion parameter

--------------------
1. Probabilistic model
a. Prior
ui, vj, tk has standard Gaussian prior
lambda has normal with 0 mean and tau-variance and tau = prod(delta_l) where delta_l ~Gamma(ac, 1)

---------------------
2. Polya Gamma augmentation
By introducing a polya-gamma variable wi, we can change the likelihood function to become Gaussian

f = exp(Ki * P_i - Wi * Pi**2/2)

where Ki = yi - mi/2 and wi ~ PG(mi, Pi)

----------------
3. Redefinition of terms + new terms
The authors introduced (instead of using C_{ik r}^(k) as in the original paper, go with the easier to understand term)


C_u[i], C_v[j], C_t[k] -> colloquially, first isolate a dimension then the C_{ik r} term is the entry
wise product of >>> Vectorview: C_{ik} =  L * [prod(ui) for all ui vector other than the specified dimension]


D_u[i], D_v[j], D_t[k] >>> Entrywise-view D_{ik r} is the sum(L * prod(all ui)) but except for the specified entry r

These are all vectors such that

phi_i = u_{ik r}^{(k)} * c_{ik r}^{(k)} + d_{ik r}^{(k)} -> This is true regardless of which dimension gets picked

--------------------
4. EM algorithm
Nice thing about the PG distribution is that its expectation has closed form
w ~ PG(b, c) => E[w] = b/2c  tanh(c/2)

For binary data, E[w] = 0.5/P tanh(P/2)
For count data, E[w] = 0.5(y+eps)/P  tanh(P/2)

a) E- step
Estimate all E[w]

b) Maximize expected log likelihood
-> Update the ui vectors as solution to the quadratic equation
-> Update the L vector also as a quadratic equation
-> For count data -> estimate the over-dispersion parameter epsilon of the negative binomial
    ???? -> newton-rhapson procedure (how)

----
May be to keep level playing field -> Not update lambda?

"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import ModuleList, ParameterList, Parameter
import math

class EM_online(nn.Module):
    def __init__(self, tensor, rank):
        super(EM_online, self).__init__()

        self.dims = tensor.dims
        self.ndim = len(self.dims)
        self.datatype = tensor.datatype
        self.tensor = tensor
        if self.datatype == "count":
            self.overdispersion_param = 1.
            self.max_count = tensor.max_count
            self.min_count = tensor.min_count

        self.train_entries = tensor.train_entries
        self.train_vals    = tensor.train_vals
        self.test_entries  = tensor.test_entries
        self.test_vals     = tensor.test_vals

        # Modify train and test vals with 0 and 1
        for i, val in enumerate(self.train_vals):
            if val == -1:
                self.train_vals[i] = 0
        for i, val in enumerate(self.test_vals):
            if val == -1:
                self.test_vals[i] = 0


        self.N = len(self.train_vals)
        self.batch_size    = 128
        self.R = rank

        self.vector_factors = []
        for dim, ncol in enumerate(self.dims):
            self.vector_factors.append(torch.ones(ncol, rank))

        self.S_diagonals = []
        # Each S^(r, k) is a diagonal matrix -> storing the diagonal only -> (ncol,) vector
        # But we have to store the similar vector (R times) for each dimension
        # Refer to the paper to assert S.shape
        # Each S^(r,k) is (ncol, ncol) matrix. Note that ncol is the number of columns of the specified dimensio
        for dim, ncol in enumerate(self.dims):
            self.S_diagonals.append(torch.zeros(ncol, rank))

        self.t_vectors = []
        # Similarly, the t_vectors have (R, ncol) for each dimension
        for dim, ncol in enumerate(self.dims):
            self.t_vectors.append(torch.zeros(ncol, rank))

        # Latent vectors
        self.L = torch.ones(rank)

    def optimize(self, max_iterations=1000, step=lambda _ : 0.01):
        """

        -------
        :return:
        """
        start          = 0
        for iteration in range(max_iterations):
            end = (start + self.batch_size) % self.N
            if end < start:
                sample_idx = list(range(end)) + list(range(start, self.N))
            else:
                sample_idx = list(range(start, end))

            batch_entries = np.take(self.train_entries, sample_idx, axis=0)
            batch_vals    = np.take(self.train_vals, sample_idx)

            # First do the expectation step
            wi_expected, c_vectors, d_vectors   = self.expectation_step(batch_entries, batch_vals)

            stepsize = step(iteration)
            # Then do the maximization step
            self.maximization_step(wi_expected, batch_entries, batch_vals, c_vectors, d_vectors, stepsize)

            if iteration in [0, 5, 10, 50] or iteration % 100 == 0:
                self.report(iteration)

    def expectation_step(self, batch_entries, batch_vals):
        """

        ----
        Computes the expectation of the wi term associated with the batch_entries

        E[wi] = 0.5/Pi tanh(Pi/2) for binary data
              = 0.5(yi + eps)/Pi  tanh(Pi/2)  for count data

        So first need to compute the Pi
        Each pi is simply sum(L * [vector])

        """
        phi_batch, c_vectors, d_vectors = self.phi_batch_compute(batch_entries) # shape = (num_samples)
        wi_expected = None
        if self.datatype == "binary":
            wi_expected = 0.5/ phi_batch * torch.tanh(phi_batch/2)
        elif self.datatype == "count":
            yi = torch.FloatTensor(batch_vals)
            wi_expected = 0.5 * (yi + self.overdispersion_param)/ phi_batch * torch.tanh(phi_batch/2)
        return wi_expected, c_vectors, d_vectors

    def maximization_step(self, wi_expected, batch_entries, batch_vals, c_vectors, d_vectors, stepsize):
        """
        :param wi_expected:
                shape = (num_samples)
        :param batch_entries:
                shape = (num_samples, ndim)
        :param batch_vals:
                shape = (num_samples)
        -----
        :return:

        Updates the hidden vectors by solving the quadratic optimization problem involving
        S and t
        -> In this case, involve the stochastic upate of S and t

        Both requires computing cik and dik
        """
        # Comute the c_ik and d_ik vectors
        # c_vectors, d_vectors = self.c_and_d_vectors_compute(batch_entries)
        ki_batch = self.ki_batch_compute(batch_vals)

        # Do stochastic update on the sufficient statistics S and t
        self.S_update_stochastic(c_vectors, wi_expected, batch_entries, stepsize)
        self.t_stochastic_update(c_vectors, d_vectors, wi_expected, ki_batch, batch_entries, stepsize)
        self.ui_vectors_update(batch_entries)


    def S_update_stochastic(self, c_vectors, wi_expected, batch_entries, stepsize):
        """
        :param c_vectors
                shape = (num_samples, ndim, R)
        :param wi_expected
                shape = (num_samples)
        :param batch_entries
                shape = (num_samples, ndim)
        :param stepsize
                a scalar

        -----
        :return:

        S is a diagonal matrix and thus we only work with the diagonal part
        S_nn^(r, k) = (1-stepsize) * S_nn ^(r, k) + stepsize * sum(c_ik^2 wi)

        For each entry in batch_entries
        -> for a particular column n with dimension k

        >>> S[dim k][column n] = (1-stepsize) * S[dim k][col n] + stepsize * c_n[entry_num, k]**2 * wi_expected[entry_num]

        """
        num_samples = len(batch_entries)
        for num, entry in enumerate(batch_entries):
            for dim, col in enumerate(entry):
                s_update = c_vectors[num, dim, :] ** 2 * wi_expected[num]
                # print(s_update)
                s_cur    = self.S_diagonals[dim][col, :]
                # print(s_cur)
                self.S_diagonals[dim][col, :] = (1-stepsize) * s_cur + stepsize * s_update

    def t_stochastic_update(self, c_vectors, d_vectors, wi_expected, ki_batch, batch_entries, stepsize):
        """

        :param c_vectors:
        :param d_vectors:
        :param wi_expected:
        :param ki_batch:
        :param batch_entries:
        :param stepsize:

        :return:
        # Assert with original paper
        >>> t[dim k][column n] += stepsize * (K[entry_num] - d_vectors[entry_num, dim, :] * wi_expected[entry_num]) * c_n[entry_num, k]
        """
        num_samples = len(batch_entries)
        for num, entry in enumerate(batch_entries):
            for dim, col in enumerate(entry):
                t_update = (ki_batch[num] - d_vectors[num, dim, :] * wi_expected[num]) * c_vectors[num, dim, :]
                t_cur    = self.t_vectors[dim][col, :]
                self.t_vectors[dim][col, :] = (1-stepsize) * t_cur + stepsize * t_update

    def ki_batch_compute(self, batch_vals):
        """

        :param batch_vals:
        -----
        :return:

        ki = yi - mi/2 where mi = 1 for binary and mi = yi + eps for count data
        """
        yi = torch.FloatTensor(batch_vals)
        if self.datatype == "binary":
            ki_batch = yi - 0.5
        else:
            ki_batch = yi/2 - self.overdispersion_param/2
        return ki_batch

    def phi_batch_compute(self, batch_entries):
        """
        :param batch_entries:
        :return: (phi_batch, c_vectors, d_vectors)

        For each observed entry
        For each dimension k
        For each involved column
        >>> c_vector[col] = self.L * [vectors from other dims] # Size R vector
        # d_vector dfined entry-wise
        >>> d_vector[col][r] = sum(self.L * [all vectors from all dims]) - self.L[r] * (prod[all vectors from all dims])
        c-vector-colletion.shape = (ndim, num_column_batch, R)
        """
        num_samples = len(batch_entries)
        inners = self.L.repeat(num_samples, 1) # Shape = (num_samples, R)
        for dim, _ in enumerate(self.dims):
            all_cols = [x[dim] for x in batch_entries]
            # all_cols = Variable(torch.LongTensor(all_cols))
            vectors  = self.vector_factors[dim][all_cols] # -> shape = (num_samples, R)
            # Get the list of vector associated with the column in the
            # specified dimension
            inners *= vectors

        phi_batch = inners.sum(dim=1) # shape = (num_samples,)
        c_vectors, d_vectors = self.c_and_d_vectors_compute(batch_entries)

        return phi_batch, c_vectors, d_vectors

    def c_and_d_vectors_compute(self, batch_entries):
        """

        :param batch_entries
               shape = (num_samples, ndim)

        -----
        :return:

        For each observed entry
        For each dimension k
        For each involved column
        >>> c_vector[dim k][col] = self.L * prod[vectors from OTHER dims] # Size R vector
        >>> d_vector[dim k][col] = sum(self.L * prod[all vectors from ALL dims]) # A scalar
        >>>                       - self.L[r] * (prod[all vectors from all dims]) # A vector

        """
        num_samples = len(batch_entries)
        c_vector = torch.ones(num_samples, self.ndim, self.R)
        d_vector = torch.ones(num_samples, self.ndim, self.R)
        involved_vectors = torch.ones(self.ndim, num_samples, self.R)

        # Get all the involved vectors from the batch_entries
        for dim, _ in enumerate(self.dims):
            all_cols = [x[dim] for x in batch_entries]
            # all_cols = Variable(torch.LongTensor(all_cols))

            vectors  = self.vector_factors[dim][all_cols] # -> shape = (num_samples, R)
            involved_vectors[dim, :, :] = vectors
        #
        for num, entry in enumerate(batch_entries):
            for dim, col in enumerate(entry):
                # All other dimensions
                other_dims = list(range(dim)) + list(range(dim+1,self.ndim))
                other_vectors_prod = torch.prod(involved_vectors[other_dims, num, :]) # shape = (R)
                c_vector[num, dim, :] = self.L * other_vectors_prod

                all_products = self.L * torch.prod(involved_vectors[:, num, :])
                d_vector[num, dim, :] = torch.sum(all_products) - all_products

        return c_vector, d_vector

    def ui_vectors_update(self, batch_entries):
        """

        :param batch_entries
        ----
        :return:

        Update ui hidden vectors by solving the quadratic maximization problem
        It has a closed form

        for each entry in batch entries
        For all the involved column (with specific dimension)

        Dimension k, column n
        u_n^(k) = S_n^(k) / t_n^(k)

        """
        num_samples = len(batch_entries)
        for num, entry in enumerate(batch_entries):
            for dim, col in enumerate(entry):
                self.vector_factors[dim][col, :] = self.S_diagonals[dim][col, :] / self.t_vectors[dim][col, :]

    def report(self, iteration):
        if iteration == 0:
            print(" iteration |  test mae  |  train mae  |")

        train_mae = self.evaluate(self.train_entries, self.train_vals)
        test_mae  = self.evaluate(self.test_entries, self.test_vals)
        print ("{:^10} {:^10} {:^10}".format(iteration, test_mae, train_mae))

    def evaluate(self, entries, vals):
        """
        ---
        :return:

        Do evaluation
        """
        test_mae = 0.
        num_entries = len(vals)
        # How to do prediction etc
        for i, entry in enumerate(entries):
            val = vals[i]
            if self.datatype == "binary":
                predict = self.binary_predict(entry)
            else:
                predict = self.count_predict(entry)
            print("actual: ", val, " predict: ", predict)
            test_mae += abs(predict - val)/num_entries
        return test_mae

    def binary_predict(self, entry):
        """

        :param entry:
        :return:

        The likelihood function is

        f = sigmoid(phi_i)
        """
        inners = torch.ones(self.R)
        for dim, col in enumerate(entry):
            inners *= self.vector_factors[dim][col, :]
        f = torch.sum(inners)
        if f > 0:
            return 1
        return 0

    def count_predict(self, entry):
        """
        :param entry:
        ---
        :return:

        f = exp(P)^yi / (1 + exp(P))^(yi + eps)

        Do maximum likelihood estimate? or weighted average
        """
        inners = torch.ones(self.R)
        for dim, col in enumerate(entry):
            inners *= self.vector_factors[dim][col, :]
        f = torch.sum(inners)

        range_vals = self.max_count - self.min_count + 1
        probs = torch.zeros(range_vals)
        for yi in range(range_vals):
            probs[yi] = torch.exp(f)/(1+torch.exp(f))**(yi+self.overdispersion_param)

        reg = torch.sum(probs)
        y   = torch.linspace(self.min_count, self.max_count + 1, steps=self.max_count - self.min_count +1) \
              * probs / reg
        return torch.sum(y)


    # NOTE: to make fair comparisons, not updating Lambda
    def S_lambda_stochastic_compute(self):
        pass

    def t_lambda_stochastic_compute(self):
        pass


class SVI(nn.Module):
    def __init__(self):
        pass


    def factorize(self):
        pass


class SVI_binary(SVI):
    def __init__(self):
        super(SVI_binary).__init__()
        pass


class SVI_count(nn.Module):
    def __init__(self):
        super(SVI_count).__init__()
        pass
