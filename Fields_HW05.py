#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:06:44 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv


def N_t(N_0,K,r,t):
    return (K*N_0)/(N_0 + (K-N_0)*np.exp(-r*t))


def dNdt(N_0,K,r,t):
    return r*N - N.T@N*(r/K)

def dNdr(N_0,K,r,t):
    n = t.size
    return ((K*N_0*t*(K-N_0)*np.exp(r*t))/(N_0*np.exp(r*t)+K-N_0)**2).reshape(n,1)

def dNdK (N_0,K,r,t):
    n = t.size
    return ((N_0**2*np.exp(r*t)*(np.exp(r*t)-1))/(K+N_0*(np.exp(r*t)-1))**2).reshape(n,1)

#def N_rr(N_0,K,r,t):
#    n = t.size
#    numA = K*N_0*t**2*(K-N_0)*np.exp(r*t)
#    numB = 2*K*N_0**2*t**2*(K-N_0)*np.exp(2*r*t)
#    denom = (K + N_0*(np.exp(r*t)-1))
#    return numA/denom**2 + numB/denom**3
#
#
#def N_rK(N_0,K,r,t):
#    numA = K*N_0*t*np.exp(r*t)
#    numB = N_0*t*(K-N_0)*np.exp(r*t)
#    numC = 2*K*N_0*t*(K-N_0)*np.exp(r*t)
#    denom = K+N_0*(np.exp(r*t)-1)
#    return numA/denom**2 + numB/denom**2 + numC/denom**3
#    
#def N_KK(N_0,K,r,t):
#    return -(2*N_0**2*np.exp(r*t)*(np.exp(r*t)-1))/(K+N_0*(np.exp(r*t)-1))**3

def LSE(N,N_0,K,r,t):
    return ((N-N_t(N_0,K,r,t))**2).sum()

def dLdr(N,N_0,K,r,t):
    term1 = (-2*N*dNdr(N_0,K,r,t)).sum()
    term2 = ((2*K**2*N_0**2*t*(K-N_0)*np.exp(-r*t))/((K-N_0)*np.exp(-r*t)+N_0)**3).sum()
    return term1 + term2

def dLdK(N,N_0,K,r,t):
    term1 = -2*N*dNdK(N_0,K,r,t).sum()
    

#def Lgrad(N,N_0,K,r,t):
#    return np.gradient(LSE(N,N_0,K,r,t)).sum()
#





data = pd.read_csv('flourbeetles.dat', sep=" ")
t = data['days'].to_numpy()
N = data['beetles'].to_numpy()


N_0=N[0]
K = 1200
r = .2
theta = [r,K]


A = np.hstack((dNdr(N_0,K,r,t),dNdK(N_0,K,r,t)))
x = N-N_t(N_0,K,r,t)

maxiter = 12

#Implement Gauss Newton Method
for i in range(maxiter):
    print("r: %.3f, K: %d" %(theta[0],np.round(theta[1])))
    theta = theta+inv(A.T@A)@(A.T@x)
    A = np.hstack((dNdr(N_0,theta[1],theta[0],t),dNdK(N_0,theta[1],theta[0],t)))
    x = N-N_t(N_0,theta[1],theta[0],t)
    















