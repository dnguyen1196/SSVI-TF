# Check the code for robust TF
- The error is increasing, why?
  + Check the normal pdf + derivatives
  + More samples? - tried, no improvements yet
  + N - N update causes error to decrease :|
  
- Do safe division with log
- The error decreases and then increases? -> constant is not 'optimal?'
- Change how I do sampling? -> 1 giant sampling?

# Implement a common interface
- Implement a common estimate_di_Di_si function
- Switch to batch processing including sampling 
- Then we need to update the probability classes
- This might be faster

# Have another implementation for suboptimal bound


# Implement function to do batch computations of derivatives


# Others
- sigma**2 vs just sigma (take note of this in current implementation)
- Implement function to compute the VLB


# poisson prediction
- Look at Rishit's code

# Questions: 
- Is it better to do approximation or sampling




