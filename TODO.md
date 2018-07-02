# Check the code for robust TF
- The error is increasing, why?
  + Check the normal pdf + derivatives
  + More samples? - tried, no improvements yet
  + N - N update causes error to decrease :|\
  
- d_cov -> it's always the same?
- w_sigma changes a lot -> increasingly (did I forget to take average?)
It seems like the problem comes from si update that gets increasingly bigger
  


# Implement a common interface

# Have another implementation for suboptimal bound


# Implement function to do batch computations of derivatives


# Others
- sigma**2 vs just sigma (take note of this in current implementation)
- Implement function to compute the VLB


# poisson prediction
- Look at Rishit's code

# Questions: 
- Is it better to do approximation or sampling






