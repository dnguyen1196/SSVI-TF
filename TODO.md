# Testing
- Focus on 0.05 of training data
- binary robust seems to do pretty poorly
why -> decreasing cov_eta to 0.001 seems to be ok
    -> reducing mean_eta only seems to make it worse as the algorithm cannot
       make progress
    -> binary prediction function is wrong? 
       -> like the cut off is 0 instaed of 1/2?
    -> Fix the covariance to true covariance and just update the mean
       -> It seems that when doing that the robust model performs pretty
          well, (overfitting actually) -> the covariance update is problematic
       -> Compares between different models?
       -> is the problem stemming from dw (CHECK if its even changing)
    
    -> round int results for count prediction 
       
    -> Optimal choice of rate constants?

- True model nll -> estimation with 

- Look at implementation and see where I could do some speed up, 
in terms of order of computation order
- Implement functions to check error of the true model
- Implement different sorts of error -> both RSME and actual error rate for binary
data

# Testing real data
 

# Other data types


# Problem with robust TF




# Draf
Robust

/home/duc/Documents/Research/SSVI-TF/venv/bin/python3 /home/duc/Documents/Research/SSVI-TF/test.py
Generating synthetic binary valued data ... 
Generating synthetic  binary valued data took:  4.0667195320129395
Evaluation for true params: 
 test_rsme | train_rsme | rel-te-err | rel-tr-err |
    0.0          0.0          0.0          0.0     
Using  0.05 of training data
Factorizing with max_iteration = 6000  fixing covariance?:  True
Tensor dimensions:  [50, 50, 50]
Optimization metrics: 
Using Ada Grad
Mean update scheme:  S
Covariance update :  N
k1 samples =  64  k2 samples =  64
eta =  1  cov eta =  0.001  sigma eta =  0.1
iteration |   time   |test_rsme |rel-te-err|train_rsme|rel-tr-err|  d_mean  |   d_cov  | test_nll | train_nll  |   dw   |
   500       541.76     0.8548     0.1827     0.8546     0.1826     2.0063       0       9134.97    1774.64      0.0    
   1000     1138.97     0.7356     0.1353     0.6597     0.1088     1.2395       0       7576.97    1256.73      0.0    
   1500     1735.86     0.6866     0.1178     0.4525     0.0512     0.7727       0       6858.24     795.7       0.0    
   2000     2332.52     0.6778     0.1148     0.2668     0.0178     0.595        0       6878.43     522.53      0.0    
   2500     2930.15     0.6784     0.115      0.1673     0.007      0.3694       0       7111.29     392.95      0.0    
   3000     3527.28     0.6756     0.1141     0.1327     0.0044     0.3923       0       7206.39     329.4       0.0    
   3500     4125.03     0.6763     0.1144     0.102      0.0026     0.3177       0       7314.13     298.04      0.0    
   4000      4722.5     0.6737     0.1135     0.0566     0.0008     0.6213       0       7352.47     274.53      0.0    
   4500     5317.97     0.6728     0.1132     0.0632     0.001      0.3049       0       7415.69     264.64      0.0    
   5000     5916.16     0.6703     0.1123     0.049      0.0006     0.331        0       7407.46     255.8       0.0    
   5500     6518.53     0.6731     0.1133     0.049      0.0006     0.2753       0       7431.92     246.59      0.0    
   6000     7129.13     0.674      0.1136     0.049      0.0006     0.3154       0       7441.38     243.09      0.0  
      
   
   


Deterministic
/home/duc/Documents/Research/SSVI-TF/venv/bin/python3 /home/duc/Documents/Research/SSVI-TF/test.py
Generating synthetic binary valued data ... 
Generating synthetic  binary valued data took:  4.040894985198975
Evaluation for true params: 
 test_rsme | train_rsme | rel-te-err | rel-tr-err |
    0.0          0.0          0.0          0.0     
Using  0.05 of training data
Factorizing with max_iteration = 6000  fixing covariance?:  True
Tensor dimensions:  [50, 50, 50]
Optimization metrics: 
Using Ada Grad
Mean update scheme:  S
Covariance update :  N
k1 samples =  64  k2 samples =  64
eta =  1  cov eta =  1  sigma eta =  1
iteration |   time   |test_rsme |rel-te-err|train_rsme|rel-tr-err|  d_mean  |   d_cov  | test_nll | train_nll  
   500       384.7       0.82      0.1681     0.8285     0.1716     1.667        0       8823.65    1720.73   
   1000      790.69     0.7143     0.1276     0.619      0.0958     1.2825       0       7590.14    1232.49   
   1500     1196.73     0.6616     0.1094     0.3688     0.034      0.5599       0       6881.46     600.95   
   2000     1603.73     0.6627     0.1098     0.1833     0.0084     0.2297       0       7821.15     285.66   
   2500     2012.08     0.6789     0.1152     0.0632     0.001      0.3604       0       11293.77    96.74    
   3000     2423.42     0.6666     0.1111     0.4345     0.0472     2.8621       0       41184.07   2977.88   
   3500     2828.83     0.7007     0.1228     0.6882     0.1184     2.3326       0       55062.57   10377.64  
   4000     3234.11     0.7114     0.1265     0.7133     0.1272     1.876        0       57610.52   11566.74  
   4500     3640.92     0.7138     0.1274     0.7178     0.1288     1.5901       0       58307.8    11771.1   
   5000     4047.89     0.7153     0.1279     0.7217     0.1302     1.4059       0       58780.82   11945.28  
   5500     4456.23     0.7154     0.128      0.7222     0.1304     1.2619       0       58911.45   12010.45  
   6000     4864.83     0.7155     0.128      0.7228     0.1306     1.1492       0       58930.82   12028.7 
   

Simple   

/home/duc/Documents/Research/SSVI-TF/venv/bin/python3 /home/duc/Documents/Research/SSVI-TF/test.py
Generating synthetic binary valued data ... 
Generating synthetic  binary valued data took:  4.28252649307251
Evaluation for true params: 
 test_rsme | train_rsme | rel-te-err | rel-tr-err |  test_nll  |  train_nll |
    0.0          0.0          0.0          0.0        22082.15     87765.55  
Using  0.05 of training data
Factorizing with max_iteration = 6000  fixing covariance?:  True
Tensor dimensions:  [50, 50, 50]
Optimization metrics: 
Using Ada Grad
Mean update scheme:  S
Covariance update :  N
k1 samples =  64  k2 samples =  64
eta =  1  cov eta =  1  sigma eta =  1
iteration |   time   |test_rsme |rel-te-err|train_rsme|rel-tr-err|  d_mean  |   d_cov  | test_nll | train_nll  |   dw   |
   500       392.73     0.8172     0.167      0.8232     0.1694     1.8354       0       8437.56     1611.7      0.0    
   1000      839.6      0.6794     0.1154     0.4996     0.0624     0.9667       0       7023.18     914.06      0.0    
   1500     1287.89     0.6696     0.1121     0.2059     0.0106     0.4814       0       9975.68     214.46      0.0    
   2000     1739.94     0.6667     0.1111     0.4454     0.0496     2.8698       0       41120.11   3253.66      0.0    
   2500     2185.47     0.7003     0.1226     0.6859     0.1176     2.3178       0       54638.62   10418.52     0.0    
   3000     2629.33     0.7127     0.127      0.7172     0.1286     1.8882       0       57926.28   11624.67     0.0    
   3500     3075.92     0.7151     0.1278     0.7222     0.1304     1.5905       0       58795.87   11981.74     0.0    
   4000     3523.72     0.7155     0.128      0.7222     0.1304     1.3944       0       58931.22   12010.28     0.0    
   4500     3971.59     0.7155     0.128      0.7228     0.1306     1.2596       0       58946.18   12028.7      0.0    
   5000      4419.9     0.7155     0.128      0.7228     0.1306     1.1452       0       58946.18   12028.7      0.0    
   5500     4868.88     0.7155     0.128      0.7228     0.1306     1.082        0       58946.18   12028.7      0.0    
   6000     5316.99     0.7155     0.128      0.7228     0.1306     0.9923       0       58946.18   12028.7      0.0    
    