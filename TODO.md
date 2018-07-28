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
- change evaluation of true model to incorporate noise?


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


Results for

No noise
All binary -> singular matrix problem
All count -> singular matrix problem
All real  -> also singular matrix problem? reset?
For no noise, they all ran for 0.01 and then die

hard noise
All binary -> also singular matrix problem
All count  -> also singular matrix problem
All real   -> obtain results, OK
Die before 0.01

ratio noise
binary -> deterministic is ok, 
       -> robust dies before 0.01
       -> simple also dies beore 0.01       
count -> deterministic is ok?
      -> simple dies 
real  -> 

