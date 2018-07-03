# Check the code for robust TF
- The error is increasing, why?
  + Check the normal pdf + derivatives
  + More samples? - tried, no improvements yet
  + N - N update causes error to decrease :|\
  
 - Try with matrix case?  N-N
 -> at least for the matrix case it makes some initial progress :| this is very
 fascinating, that means there is some difference between the formulation
 for matrix and tensor?
 -> it stops making progress -> try with constant step size (simple sgd) -> simple sgd is not gonna cut it
 size or something that is less severe than ada grad 
 -> but it seems that the d_mean drop like crazy :| -> may be the 
 gradient is really high and first and therefore this makes it dump down 
 the gradient later -> not use adagrad but adadelta? -> use a windows of recent 
 gradient instead of accummulating sum of gradient
 
- Why is the error increasing? seems monotonically too :|
- obviously d_cov highly depends on d_mean  
- covariance matrix becomes indefinite -> why? and why does the d_cov so high 
regardless of the cov_eta?
- and why does the error increases regardless of the sign of di?

TODO:
- Check the formulation once again -> make sure it's 100% correct
- Also look at the update formula + the generic update formula too :|
- Look at the previous commits and see any difference in formulation that could 
have explained the success

# Implement a common interface

# Have another implementation for suboptimal bound


# Implement function to do batch computations of derivatives


# Others
- sigma**2 vs just sigma (take note of this in current implementation)
- Implement function to compute the VLB
- Report RSME

# poisson prediction
- Look at Rishit's code
- Basically doing sampling -> algebraic transformation to avoid looping -> saves cost

# Questions: 
- Is it better to do approximation or sampling






