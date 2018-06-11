# Check the code for robust TF
- first look at the derivations again to be sure
- then look at the implementation again, may be it's better to have separate 
samplings to compute the different expectations

# - Implement separation between H-MC-SSVI and S-MC-SSVI and N-MC-SSVI (mode)
- So there will be different options to do the update formula
- Implement as functions and chosen at runtime
- 

# Fix poisson prediction
- Look at why the errors shoot up after reaching relatively low error rate
- Look at the actual prediction that the algorithm outputs
- Send some report to Roni

# Stopping condition check
- Keeps track of changes in norm of mean vectors and changes in frobenius norm
of matrix
 