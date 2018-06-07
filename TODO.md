# - Implement separation between H-MC-SSVI and S-MC-SSVI and N-MC-SSVI (mode)


# Fix poisson prediction
- Check why the covariance is 0 (check the eigenvalue lol), it seems random, either
ok or not ok :|
- why does the error seem to be increasing instead of decreasing???
may be the gradient is not correct, check the tensor to see if it agrees
- The problem doesnt seem to be from initializing the int data, when I tried to spread out the 
mean of the distribution, the same problem persists
- tried computing the posterior parameter explicitly, same problem persists
It seems that the accummulate sum is yugee

- initializing from all mean_q = 1, so some accummulate inner product is pretty big
may be it's the way I do the rounding
- It seems that the value for f keep blowing up  :|
- Make sure that the simple model works first, that the count is the round of the inner product
of the mean vectors
- it seems like he vectors keep diverging away from the supposed value