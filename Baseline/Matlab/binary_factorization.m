clear all; close all;
% load facebook_xiid
addpath('BinaryTensorFactorization');

load binary_synthetic_tensor

R=10;
numiters=60;
fraction=5;% ratio between number of zeros and ones in testing data
for k=1:3 
    N(k) = max(id{k}); 
end

trainfraction=0.9;%90 percent as training data
isbatch=0;% 1: batch gibbs; 0: online gibbs
batchsize=floor(length(id{1})*trainfraction/10);%9/1 training/test split, batch size is 10 percent of training data

[U lambda pr eva time_trace] = BTF_OnlineGibbs(N,xi,id,R,batchsize,numiters,isbatch,fraction,trainfraction);