# Testing
- Implement function to perform learning curve investigation
-> 0.8 data train data, 0.2 test -> then use 20%, 40%, ..., 100% of training data
and see how it performs 
-> Fix a constant number of iterations?
-> output everything to a file

-> Check problem with generating data?

-> Re-implement main to get better flexibility


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





