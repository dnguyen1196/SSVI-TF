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


hard noise for real -> 500
hard noise for binary -> 0.5
hard noise for count -> 1

# extra implementation
- Output to specified files instead of waiting on stddout

# diagonal covariance


# Testing real data
 

# Other data types
- Robust count 3D -> why sigma update results in nan? (really interesting
and weird)

# Problem with robust TF
- The covariance becoming non-positive definite



binary deterministic
count deterministic
real simple
binary simple
count simple
real robust
binary robust
count robust


-p batch -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER --wrap="python learning_curve.py -d real -m deterministic" 


full_cov_jul_25_hard_noise/
full_cov_jul_25_no_noise/
full_cov_jul_25_ratio_0.1/

full_cov_jul_25_hard_noise
full_cov_jul_20_noise_0.1  
full_cov_jul_25_no_noise
