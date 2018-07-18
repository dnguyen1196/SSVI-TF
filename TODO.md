# Testing
- noise at 100? is this too much? try with different noise level?
- might have to increase number of iterations :|
- Look at implementation and see where I could do some speed up, 
in terms of order of computation
- add noise in terms of percentage

# diagonal covariance
- the scale becomes < 0 (cov_batch contains negative values? wy)
- Line 222 of interface -> what is it doing

# Testing real data
 

# Other data types
- Robust count 3D -> why sigma update results in nan? (really interesting
and weird)

# Problem with robust TF
- Need to fix robust count 
- Robust binary and real seem to be working ? => just need more data to look at

-> May be if the norm of covgrad is too big -> then ignore this update for now?
Capping the covariance update seems to work for now? -> Need to figure out exactly what's up

-> Removing the negative eigenvalues dosent seem to do the trick 
-> but capping the covariance seems to produce very shitty results?
-> diagonal covariance seems to be a good choice to deal with covariance problem
in robust model

-> Check simple prediction for binary -> bound is 1/2


# Questions: 






