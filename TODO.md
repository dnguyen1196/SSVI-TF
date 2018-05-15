# step-size for covariance matrix 
- checked

# Add update formula for covariance parameter
- checked

# Identify places where it can run out of memories 
(think about diagonal matrix formulation vs full matrix formulation)
(may be it's bound to happen?)

 1000 X 1000 X 1000
 
 3000 factors * (D mean + D^2 covariance)
 if D = 20, 400 * 3000 = 1 200 000 (not that much ay)
 
# Implement binary and poisson likelihood
- Start with distribution.py
- Implement link function


# Fix the synthesize function (right now still checking for duplicates
which takes a lot of time)