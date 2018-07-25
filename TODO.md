# Testing
- Look at implementation and see where I could do some speed up, 
in terms of order of computation order
- Report accuracy for binary prediction instead of RSME
- The robust model is not stable :| -> cov_eta is the problem 
-> cov_eta 0.001, but this might

Look at the results I have it seems that after some iterations, at one point
the d_cov shoot up :| really mysteriously. Let's run 50 x 3 for a while and see

- Implement functions to check error of the true model
- Implement different sorts of error -> both RSME and actual error rate for binary
data

# extra implementation
- Generate data using both fixed and variable stddev
- Learning curve -> smaller portion of data 2%, 5%, 10%, 15%, 20%, 


# diagonal covariance


# Testing real data
 

# Other data types
- Robust count 3D -> why sigma update results in nan? (really interesting
and weird)

# Problem with robust TF
- The covariance becoming non-positive definite







