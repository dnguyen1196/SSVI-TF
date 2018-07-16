# Testing
- noise at 100? is this too much
- Look at implementation and see where I could do some speed up, 
in terms of order of computation
- diagonal covariance?


# Other data types
- Robust count 3D -> why sigma update results in nan? (really interesting
and weird)

# Problem with robust TF
- Need to fix robust count 
- Robust binary and real seem to be working ? => just need more data to look at

Look at derivative for poisson case

-> see vjk norm and phi_snd/phi?
-> May be if the norm of covgrad is too big -> then ignore this update for now?
Capping the covariance update seems to work for now? -> Need to figure out exactly what's up


-> Removing the negative eigenvalues dosent seem to do the trick 
-> but capping the covariance seems to produce very shitty results?
-> diagonal covariance seems to be a good choice to deal with covariance problem
in robust model


-> Check simple prediction for binary -> bound is 1/2
-> Data generation is faulty for binary lmao! no?

why would there be a difference in the way i initialize data?

# Questions: 
- Complete main.py
-> option on the command line that can be ported an run on tufts
cluster





