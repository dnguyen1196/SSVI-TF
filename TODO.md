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
-more smaples?
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


# other data types
- implement sample function

# Others
- sigma**2 vs just sigma (take note of this in current implementation)
- Implement function to compute the VLB
- Report RSME

# poisson prediction
- Look at Rishit's code
- Basically doing sampling -> algebraic transformation to avoid looping -> saves cost

# Questions: 
- Is it better to do approximation or sampling






