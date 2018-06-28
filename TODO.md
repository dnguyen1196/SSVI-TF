# Implement a common interface
- pSigma vs pSigma inverse
- Posterior_diag_covariance class

# Have another implementation for suboptimal bound
- Check if there is a closed form update for the case of complete conditional 


# Check the code for robust TF
- Reimplement the gradient computation according to the new formulas
- First start by writing down what i need to estimate (looks like a lot 
can be reused)
- Then implement

# Implement function to do batch computations of derivatives


# 
- sigma**2 vs just sigma (take note of this in current implementation)

# Additional implementation
- Implement function to compute the VLB


# Fix poisson prediction
- Look at the actual prediction that the algorithm outputs
  File "/home/duc/Documents/Research/SSVI-TF/SSVI/SSVI_TF_d.py", line 463, in compute_expected_count
    return np.sum(np.multiply(probs, np.array(k_array)))/sum_probs
RuntimeWarning: invalid value encountered in double_scalars
The above problem seems to happen with "big" stepsize

# numerical approximation vs sampling
- Is it better to do approximation or sampling




