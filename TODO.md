# Additional implementation
- Why is the NLL increasing instead of decreasing
- And while the NLL is increasing, why does the error decreasing, 

# Have another implementation for suboptimal bound

# Check the code for robust TF
- Reimplement the gradient computation according to the new formulas

# Fix poisson prediction
- Look at the actual prediction that the algorithm outputs
  File "/home/duc/Documents/Research/SSVI-TF/SSVI/SSVI_TF_d.py", line 463, in compute_expected_count
    return np.sum(np.multiply(probs, np.array(k_array)))/sum_probs
RuntimeWarning: invalid value encountered in double_scalars
The above problem seems to happen with "big" stepsize

# approximation vs sampling
- Is it better to do approximation or sampling
