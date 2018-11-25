# Bayes Poisson Factorization

Most important is to prepare the data file

See BayesPoissonFactor/demo_parallel.m

xi_id.mat which has two variables

id = (num_entries, dimension) stored into a 1 x 4 cell object
Each cell is an (num_entries,) vector that contains the integer index/coordinate
of the observed entries

xi = (num_entries,1) vector that contains the integer valued stored in the entries


# Binary Tensor Factorization

Look into /BinaryTensorFactorization/demo_gibbs_large.m

Since it loads duke_xiid.m but it contains co_id and co_xi but the algorithm
only uses id and xi. Therefore, it looks like we can follow the same procedure
as above as we prepare that file.

-> id 
-> xi


