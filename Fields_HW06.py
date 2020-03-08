#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:38:54 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, relfreq, multivariate_normal

x = np.vstack((np.zeros((162,1)),np.ones((267,1)),2*np.ones((271,1)),
               3*np.ones((185,1)), 4*np.ones((111,1)),
               5*np.ones((61,1)), 6*np.ones((27,1)), 7*np.ones((8,1)),
               8*np.ones((3,1)), 9*np.ones((1,1))))

def pihat(z):
    n = z.size
    return np.sum(z/n)

def l1hat(x,z):
    return np.sum(x-z*x)/np.sum(1-z)

def l2hat(x,z):
    return np.sum(z*x)/np.sum(z)

def f(x,pi,l1,l2):
    return (1 - pi)*poisson.pmf(x,l1)+pi*poisson.pmf(x,l2)

def E(x,pi,l1,l2):
    a = pi*poisson.pmf(x,l2)
    b = (1 - pi)*poisson.pmf(x,l1)+pi*poisson.pmf(x,l2)
    return a/b

maxiter = 1000
pi = .1
l1 = 2
l2 = 2

for i in range(maxiter):
    print("pi: %f, l1: %f, l2: %f"%(pi, l1, l2))
    z = E(x,pi,l1,l2)
    pi = pihat(z)
    l1 = l1hat(x,z)
    l2 = l2hat(x,z)
    
xx = np.linspace(0,9,1000)
plt.plot(xx,f(xx,pi,l1,l2))
res = relfreq(x, numbins=10)
xx = np.arange(10)
plt.plot(xx,res[0])
plt.legend(('Fitted Prob','Relative Freq'))
print()
print("The probability of x=10: %f" %f(10,pi,l1,l2))
print()

"""
Problem 2
"""

    
data = pd.read_csv('data_mvnorm2mix.csv',sep=',',header=None).to_numpy()


def pihat(z):
    n = z.size
    return np.sum(z/n)

def mu1hat(z,x):
    n = z.size
    l = []
    for i in range(n):
        l.append(((1-z[i])*x[i])/(n-np.sum(z[i])))
    return np.array(l).sum(axis=0).reshape(2,1)

def mu2hat(z,x):
    n = z.size
    l = []
    for i in  range(n):
        l.append((z[i]*x[i])/z[i])
    return np.array(l).sum(axis=0).reshape(2,1)

def sig1hat(z,x,mu1):
    n= z.size
    l = []
    for i in range(n):
        l.append(((1-z[i])*(x[i]-mu1)@(x[i]-mu1).T)/(n-z[i]))
    return np.array(l).sum(axis=0)

def sig2hat(z,x,mu2):
    n = z.size
    l = []
    for i in range(n):
        l.append((z[i]*(x[i]-mu2)@(x[i]-mu2).T)/z[i])
    return np.array(l).sum(axis=0)



def fx(z,x,pi,mu1,mu2,sig1,sig2):
    return (1-pi)*multivariate_normal.pdf(x,mean = mu1.reshape(2,),cov=sig1)+\
        pi*multivariate_normal.pdf(x,mean = mu2.reshape(2,),cov=sig2)
        
def Ez(z,x,pi,mu1,mu2,sig1,sig2):
    return pi*multivariate_normal.pdf(x,mean = mu2.reshape(2,),cov=sig2)/fx(z,x,pi,mu1,mu2,sig1,sig2)

maxiter = 100
pi = .8
mu1 = np.array([[1.5],[2.5]])
mu2 = np.array([[1.5],[2.5]])
sig1 = np.array([[3.5,4.5],[.5,.5]])
sig2 =  np.array([[3.5,4.5],[.5,.5]])

for i in range(maxiter):
    print("pi: ",pi,"mu1: ", mu1,"mu2: ", mu2)
    z = Ez(z,data,pi,mu1,mu2,sig1,sig2)
    pi = pihat(z)
    mu1 = mu1hat(z,data)
    mu2 = mu2hat(z,data)
    sig1 = sig1hat(z,data,mu1)
    sig2 = sig2hat(z,data,mu2)