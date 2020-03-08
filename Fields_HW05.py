#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:06:44 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv
from numpy import *


def N_t(N_0,K,r,t):
    return (K*N_0)/(N_0 + (K-N_0)*np.exp(-r*t))

#Functions for part a
def dNdt(N_0,K,r,t):
    return r*N - N.T@N*(r/K)

def dNdr(N_0,K,r,t):
    n = t.size
    return ((K*N_0*t*(K-N_0)*np.exp(r*t))/(N_0*np.exp(r*t)+K-N_0)**2).reshape(n,1)

def dNdK (N_0,K,r,t):
    n = t.size
    return ((N_0**2*np.exp(r*t)*(np.exp(r*t)-1))/(K+N_0*(np.exp(r*t)-1))**2).reshape(n,1)

#Functions for part b
def LSE(N,N_0,K,r,t):
    return ((N-N_t(N_0,K,r,t))**2).sum()

def dLdr(N,N_0,K,r,t):
    term1 = ((-2*K*N_0*t*N*(K-N_0)*exp(-r*t))/(N_0+(K-N_0)*exp(-r*t))**2)
    term2 = ((2*K**2*N_0**2*t*(K-N_0)*np.exp(-r*t))/((K-N_0)*np.exp(-r*t)+N_0)**3)
    return term1.sum() + term2.sum()

def dLdK(N,N_0,K,r,t):
#    term1 = -2*N*dNdK(N_0,K,r,t).sum()
    term1 = (-2*N_0**2*N*exp(r*t)*(exp(r*t)-1))/(K+N_0*(exp(r*t)-1))**2
    term2 = (2*K*N_0**3*exp(2*r*t)*(exp(r*t)-1))/(K+N_0*(exp(r*t)-1))**3
    return term1.sum() + term2.sum()

def L_rr(N,N_0,K,r,t):
    term1 = (2*K*N_0*t**2*N*(K-N_0)*exp(r*t)*(-K+N_0*exp(r*t)+N_0))/(K+N_0*(exp(r*t)-1))**3
    term2 = (2*K**2*N_0**2*t**2*(K-N_0)*exp(2*r*t)*(2*K-N_0*exp(r*t)-2*N_0))/(K+N_0*exp(r*t)-N_0)**4
    return term1.sum() + term2.sum()

def L_kk(N,N_0,K,r,t):
    term1 = (4*N_0**2*N*exp(r*t)*(exp(r*t)-1))/(K+N_0*(exp(r*t)-1))**3
    term2 = (2*N_0**3*exp(2*r*t)*(exp(r*t)-1)*(-2*K+N_0*exp(r*t)-N_0))/(K+N_0*exp(r*t)-N_0)**4
    return term1.sum() + term2.sum()
    
def L_rk(N,N_0,K,r,t):
    term1 = (-2*N_0**2*t*N*exp(r*t)*(2*K*exp(r*t)-K-N_0*exp(r*t)+N_0))/(K+N_0*(exp(r*t)-1))**3
    term2 = (2*K*N_0**3*t*exp(2*r*t)*(3*K*exp(r*t)-2*K-2*N_0*exp(r*t)+2*N_0))/(K+N_0*exp(r*t)-N_0)**4
    return term1.sum() + term2.sum()


data = pd.read_csv('flourbeetles.dat', sep=" ")
t = data['days'].to_numpy()
N = data['beetles'].to_numpy()


N_0=N[0]
K = 1200
r = .6
theta = [r,K]


A = np.hstack((dNdr(N_0,K,r,t),dNdK(N_0,K,r,t)))
x = N-N_t(N_0,K,r,t)

maxiter = 12

#Implement Gauss Newton Method
print()
print('Gauss-Newton Method')
for i in range(maxiter):
    print("r: %.3f, K: %d" %(theta[0],np.round(theta[1])))
    theta = theta+inv(A.T@A)@(A.T@x)
    A = np.hstack((dNdr(N_0,theta[1],theta[0],t),dNdK(N_0,theta[1],theta[0],t)))
    x = N-N_t(N_0,theta[1],theta[0],t)
    
t = data['days'].to_numpy()
N = data['beetles'].to_numpy()


N_0=N[0]
K = 1200
r = .3
theta = [r,K]
maxiter = 20

grad = np.array([dLdr(N,N_0,K,r,t),dLdK(N,N_0,K,r,t)])
H = np.array([[L_rr(N,N_0,K,r,t),L_rk(N,N_0,K,r,t)],[L_rk(N,N_0,K,r,t),L_kk(N,N_0,K,r,t)]])

    
#Implement Newton Raphson
print()
print('Newton-Raphson Method')
for i in range(maxiter):
    print("r: %.3f, K: %d" %(theta[0],np.round(theta[1])))
#    print('grad: ', grad)
#    print('hessian: ', H)
    theta = theta-inv(H)@grad
    grad = np.array([dLdr(N,N_0,theta[1],theta[0],t),dLdK(N,N_0,theta[1],theta[0],t)])
    H = np.array([[L_rr(N,N_0,theta[1],theta[0],t),L_rk(N,N_0,theta[1],theta[0],t)],\
                   [L_rk(N,N_0,theta[1],theta[0],t),L_kk(N,N_0,theta[1],theta[0],t)]])

    
#part C: extra
import sympy as sp

#sigma=1
#K=1200
#N0=2
#r=.2

#symbols = sp.symbols
#r, t, K, N, N0 = symbols('r t K N N0')
#
#LSE = (sp.log(N)-sp.log((K*N0)/(N0 + (K-N0)*sp.exp(-r*t))))**2
#dLdr = sp.diff(LSE,r,1)
#dldK = sp.diff(LSE,K,1)


def dLdr(N,N0,K,r,t):
    x=-2*t*(K - N0)*(log(N) - log(K*N0/(N0 + (K - N0)*exp(-r*t))))*exp(-r*t)/(N0 + (K - N0)*exp(-r*t))
    return x.reshape(10,1)
    
def dLdk(N,N0,K,r,t):
    x=-2*(N0 + (K - N0)*exp(-r*t))*(-K*N0*exp(-r*t)/(N0 + (K - N0)*exp(-r*t))**2 + N0/(N0 + (K - N0)*exp(-r*t)))*(log(N) - log(K*N0/(N0 + (K - N0)*exp(-r*t))))/(K*N0)
    return x.reshape(10,1)
t = data['days'].to_numpy()
N = data['beetles'].to_numpy()

print()

N_0=2
K = 900
r = .15
theta = [r,K]
A = np.hstack((dLdr(N,N_0,K,r,t),dLdk(N,N_0,K,r,t)))
x = log(N)-log(N_t(N_0,K,r,t))

print()
print('C: Gauss-Newton Method')
for i in range(20):
    print("r: %.3f, K: %d" %(theta[0],np.round(theta[1])))
    theta = theta+inv(A.T@A)@(A.T@x)
    A = np.hstack((dLdr(N,N_0,theta[1],theta[0],t).reshape(10,1), dLdk(N,N_0,theta[1],theta[0],t).reshape(10,1)))
    x = log(N)-log(N_t(N_0,theta[1],theta[0],t))