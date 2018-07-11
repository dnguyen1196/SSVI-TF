# Other data types
- Implement function to do estimates by sampling -> both for bernoulli (with noise)
and poisson (<- most important?)
- First figure out the predictive integral

- Thought experiment: running simple model without noise prediction? Is it better than
deterministic model? ->
- Simple binary with sampling prediction -> same error forever?? wy

# Problem with robust TF
- Need to fix robust count 
- Robust binary and real seem to be working ? => just need more data to look at

Look at derivative for poisson case

- wrong derivative
Generating synthetic data ... 
Generating synthetic data took:  0.30700016021728516
max count is:  18
di:  466.776431359
Di:  7.56139410722e+89 -> This is the problem, Di why is it like this?
-> may be some thing cancel out -> causes this problem?

-> see vjk norm and phi_snd/phi?

-> May be if the norm of covgrad is too big -> then ignore this update for now?
Capping the covariance update seems to work for now? -> Need to figure out exactly what's up

-> It seems that for the poisson robust model the covariance update
is highly unstable -> recommend using the diagonal update?

-> Removing the negative eigenvalues dosent seem to do the trick -> but capping the covariance 
seems to produce very shitty results?
-> diagonal covariance?


-> The problem with doing batch computation -> how to handle individual 
entry special cases
-> Keep in mind that it must still work for binary deterministic and
binary simple!


-> what numpy does with the warning/ does it assign infinity? or 0



# Questions: 
- Is it better to do approximation or sampling






