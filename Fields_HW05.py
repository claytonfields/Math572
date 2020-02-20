#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:06:44 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv
from numpy import exp


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
    
#def L_rk(N,N_0,K,r,t):
#    term1 = (4*N_0**2*N*exp(r*t)*(exp(r*t)-1))/(K+N_0*(exp(r*t)-1))**3
#    term2 = (2*N_0**3*exp(2*r*t)*(exp(r*t)-1)*(-2*K+N_0*exp(r*t)-N_0))/(K+N_0*exp(r*t)-N_0)**4
#    return term1.sum() + term2.sum()

def L_rk(N,N_0,K,r,t):
    term1 = (-2*N_0**2*t*N*exp(r*t)*(2*K*exp(r*t)-K-N_0*exp(r*t)+N_0))/(K+N_0*(exp(r*t)-1))**3
    term2 = (2*K*N_0**3*t*exp(2*r*t)*(3*K*exp(r*t)-2*K-2*N_0*exp(r*t)+2*N_0))/(K+N_0*exp(r*t)-N_0)**4
    return term1.sum() + term2.sum()

#def Lgrad(N,N_0,K,r,t):
    
#    return np.gradient(LSE(N,N_0,K,r,t)).sum()
#





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
sigma=1
K=1200
N0=2
r=.2

def dldr(N,N0,r,K,t,sigma):
    l = []
    for i in range(N.size):
        f1 = -(1/sigma**2)*(np.log(N[i])-np.log(K)-np.log(N0)+np.log(N0+(K-N0)*np.exp(-r*t[i])))
        f2 = (-t[i]*(K-N0)*np.exp(-r*t[i]))/(N0+(K-N0)*np.exp(-r*t[i]))
        l.append(f1*f2)
    return np.asarray(l).reshape(N.size,1)

def dldK(N,N0,r,K,t,sigma):
        f1
    
    
"""
Problem 02: From example 4.2
"""
def ENcc(nc,pc,pi,pt):
    return (nc*pc**2)/(pc**2+2*pc*pi+2*pc*pt)

def ENci(nc,pc,pi,pt):
    return (2*nc*pc*pi)/(pc**2+2*pc*pi+2*pc*pt)

def ENct(nc,pc,pi,pt):
    return (2*nc*pc*pt)/(pc**2+2*pc*pi+2*pc*pt)

def ENii(ni,pi,pt):
    return (ni*pi**2)/(pi**2+2*pi*pt)

def ENit(ni,pi,pt):
    return (2*ni*pi*pt)/(pi**2+2*pi*pt)

def p_c(ncc,nci,nct):
    return (2*ncc+nci+nct)/(2*622)

def p_i(nii,nit,nci):
    return (2*nii+nit+nci)/(2*622)

def p_t(nit,nct,ntt):
    return (2*ntt+nct+nit)/(2*622)

nc = 85; ni = 196; nt = ntt = 341
n = nc + ni +nt
pc = pi = pt = .333333

t=0
print()
for i in range(20):
    #E step
    
    foo  = [pc,pi]
    ncc = ENcc(nc,pc,pi,pt)
    nci = ENci(nc,pc,pi,pt)
    nct = ENct(nc,pc,pi,pt)
    nii = ENii(ni,pi,pt)
    nit = ENit(ni,pi,pt)
    #Mstep
    pc = p_c(ncc,nci,nct)
    pi = p_i(nii,nit,nci)
    pt = p_t(nit,nct,ntt)
    
    dc = abs((pc-.07084)/(foo[0]-.07084))**2
    di = abs((pi-.18874)/(foo[1]-.18874))**2
    
    print(t)
    print(pc)
    print(pi)
    print(pt)
    print('Dc: ', dc)
    print('Di: ', di)
    print()
    t+=1




