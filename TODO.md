# Check implementation
- may be we don't need to sample for other vectors!!!


# Additional implementation
- Decreasing step size
- keep track of the VLB + NLL

# Check the code for robust TF
- Reimplement the gradient computation according to the new formulas

# Fix poisson prediction
- Look at why the errors shoot up after reaching relatively low error rate 
(at least the error is not blowing up now :|)
- Implement the more 'accurate' formulas for prediction lmao
- Look at the actual prediction that the algorithm outputs

  File "/home/duc/Documents/Research/SSVI-TF/SSVI/SSVI_TF_d.py", line 463, in compute_expected_count
    return np.sum(np.multiply(probs, np.array(k_array)))/sum_probs
RuntimeWarning: invalid value encountered in double_scalars

probability becomes 0 :| check

- Send some report to Roni

# approximation vs sampling
- Is it better to do approximation or sampling
