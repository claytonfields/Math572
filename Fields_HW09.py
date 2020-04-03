#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:50:55 2020

@author: claytonfields
"""

import numpy as np
from scipy.stats import norm

#def S(r,sigma,T,s0):
    
## INITIAL VALUES
S0 = 50
K = 52
sigma = 0.5
T = 30
r = 0.05
n = 100
m = 10

## BACKGROUND: MC ESTIMATES (EUROPEAN CALL OPTION)
mu_mc_e = np.zeros((m,1))
for j in range(m):
      ST = S0*np.exp((r-(sigma**2)/2)*T/365 + sigma*norm.rvs(size=1000)*np.sqrt(T/365))
      C = np.zeros((n,1))
      for i in range(n):
      	    C[i] =np.exp(-r*T/365)*max(0,ST[i] - K)
      
      mu_mc_e[j] = np.mean(C)

se_mc_e = np.std(mu_mc_e)/np.sqrt(m)    
Muhat = mu_mc_e.mean()
print(Muhat)

# MC ESTIMATES (ASIAN ARITHMETIC AND GEOMETRIC CALL OPTION)
#mu_mc = np.zeros((100,1))
theta_mc = np.zeros((100,1))
mu_mc = A

for j in range(m):
#calculate MC estimate of A and theta
#    A = np.zeros((n,1))
    theta = np.zeros((n,1))
    A = []
    for i in range(n):
#        ST= np.zeros((K-1,1))
        ST = []
        ST.append(S0)
        for k in range(1,T):
#             ST[k]= ST[k-1]*np.exp(((r-(sigma**2)/2)/365) + norm.rvs(size=1)/np.sqrt(365))
             ST.append(np.asscalar(ST[k-1]*np.exp(((r-(sigma**2)/2)/365) + sigma*norm.rvs(size=1)/np.sqrt(365))))
#        A[i] = np.exp(-r*T/365)*max(0,np.mean(ST) - K)
        A.append(np.exp(-r*T/365)*max(0,np.mean(ST) - K))
        theta[i] = np.exp(-r*T/365)*max(0,np.exp(np.mean(np.log(ST))) - K)
#    mu_mc.append(np.mean(A))
#    mu_mc[j] = np.mean(A)
    theta_mc[j]=np.mean(theta)
np.mean(A)
#theta_mc = np.mean(theta_mc)
## ANALYTIC SOLUTION (GEOMETRIC MEAN)
N = T
c3 = 1 + 1/N
c2 = sigma*((c3*T/1095)*(1 + 1/(2*N)))**.5
c1 = (1/c2)*(np.log(S0/K) + (c3*T/730)*(r - (sigma**2)/2) +
       (c3*(sigma*2)*T/1095)*(1 + 1/(2*N)))
theta = S0*norm.cdf(c1)*np.exp(-T*(r + c3*(sigma**2)/6)*(1 - 1/N)/730) - K*norm.cdf(c1-c2)*np.exp(-r*T/365)

## CONTROL VARIATE
mu_cv=mu_mc-1*(theta_mc-theta)

## OUTPUT
print(np.std(mu_mc))  #STANDARD DEVIATION FOR ORDINARY APPROACH
print(np.std(mu_cv)) #STANDARD DEVIATION FOR CONTROL VARIATE APPROACH








