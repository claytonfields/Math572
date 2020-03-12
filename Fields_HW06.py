#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:38:54 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

maxiter = 10
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
    return np.sum(z)/n

def mu1hat(z,x):
    n = z.size
    l = []
    for i in range(n):
        l.append(((1-z[i])*x[i].T))
    num = np.sum(l,axis=0).reshape(2,1)
    den = (n-np.sum(z))
    return num/den

def mu2hat(z,x):
    n = z.size
    l = []
    for i in  range(n):
        l.append((z[i]*x[i].T))
    num = (np.sum(l,axis=0).reshape(2,1))
    den = np.sum(z)
    return num/den

def sig1hat(z,x,mu1):
    n= z.size
    l = []
    for i in range(n):
        l.append(((1-z[i])*(x[i].reshape(2,1)-mu1.reshape(2,1))@(x[i].reshape(2,1)-mu1).T))
    num = np.sum(l,axis=0)
    den = (n-np.sum(z))
    return num/den

def sig2hat(z,x,mu2):
    n = z.size
    l = []
    for i in range(n):
        l.append((z[i]*(x[i].reshape(2,1)-mu2)@(x[i].reshape(2,1)-mu2).T))
    num = np.array(l).sum(axis=0)
    den = np.sum(z)
    return num/den



def fx(x,pi,mu1,mu2,sig1,sig2):
    term1 = (1-pi)*multivariate_normal.pdf(x,mean = mu1.reshape(2,),cov=sig1)
    term2 =  pi*multivariate_normal.pdf(x,mean = mu2.reshape(-1,),cov=sig2)
    return term1 + term2
       
        
def Ez(x,pi,mu1,mu2,sig1,sig2):
    return pi*multivariate_normal.pdf(x,mean = mu2.reshape(-1,),cov=sig2)/fx(x,pi,mu1,mu2,sig1,sig2)

def L(data,pi,mu1,mu2,sig1,sig2):
    n = z.size
    l = []
    for i in range(n):
        term1 = (1-z[i])*np.log(multivariate_normal.pdf(data[i],mean = mu1.reshape(-1,),cov=sig1))
        term2 = z[i]*np.log(multivariate_normal.pdf(data[i],mean = mu2.reshape(-1,),cov=sig2))
        term3 = (1-z[i])*np.log(1-pi)+z[i]*np.log(pi)
        l.append(term1+term2+term3)
    return np.sum(l)
        
maxiter = 25
pi = .6
mu1 = np.array([[1],[2]])
mu2 = np.array([[2],[4]])
sig1 = np.array([[2,1],[3,8]])
sig2 =  np.array([[1,-1],[-1,6]])
loglist = []

for i in range(maxiter):
    print("pi: ",pi,"mu1: ", mu1,"mu2: ", mu2)
#    print("pi: ",pi,"sig1: ", sig1,"sig2: ", sig2)
    z = Ez(data,pi,mu1,mu2,sig1,sig2)
    z = z.reshape(800,1)
    pi = pihat(z)
    mu1 = mu1hat(z,data)
    mu2 = mu2hat(z,data)
    sig1 = sig1hat(z,data,mu1)
    sig2 = sig2hat(z,data,mu2)
    loglist.append(L(data,pi,mu1,mu2,sig1,sig2))
 
domain = np.linspace(1,25,25)
plt.figure()
plt.plot(domain,loglist)
plt.title("Loglikelihood vs iteration")

#Part C
def dis1(x,mu1,sig1):
    return multivariate_normal.pdf(x,mean = mu1.reshape(2,),cov=sig1)

def dis2(x,mu2,sig2):
    return multivariate_normal.pdf(x,mean = mu2.reshape(-1,),cov=sig2)

x1b1list = []
x1b2list = []
x2b1list = []
x2b2list = []

for i in range(800):
    if dis1(data[i],mu1,sig1)>dis2(data[i],mu2,sig2):
        x1b1list.append(data[i,0])
        x2b1list.append(data[i,1])
    else:
        x1b2list.append(data[i,0])
        x2b2list.append(data[i,1])
plt.figure()
plt.scatter(x1b1list,x2b1list)
plt.scatter(x1b2list,x2b2list)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(("Distribution 1","Distribution 2"))
plt.title("Observation Classification")


#Part D
xmin  = ymin =  -5
xmax = ymax = 5
xx,yy = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
z = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        temp1 = xx[i,j]
        temp2 = yy[i,j]
        xystack = np.hstack((temp1,temp2))
        z[i,j] = fx(xystack,pi,mu1,mu2,sig1,sig2)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx,yy,z)
plt.title('Surface plot of fitted function')
#plt.plot_surface(xx,yy,z)

