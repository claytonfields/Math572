#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:33:24 2020

@author: claytonfields
"""

import numpy as np
from scipy.stats import norm,uniform, poisson,lognorm
import matplotlib.pyplot as plt

def slash(x):
    if x==0:
        return 1/(np.sqrt(2*np.pi))
    else:
        return (1-np.exp(-x**2/2))/(x**2*np.sqrt(2*np.pi))
    

y = norm.rvs(size=100000)/uniform.rvs(size=100000)

denom = []
for y_i in y:
    denom.append(norm.pdf(y_i)/slash(y_i))
denom = np.sum(denom)

def w(x,denom):
    return (norm.pdf(x)/slash(x))/denom
wvals = []
for y_i in y:
    wvals.append(np.asscalar(w(y_i,denom)))

xvals = np.random.choice(y,size=5000,replace=True,p=wvals)
fig = plt.figure()
axes1 = fig.add_subplot(1,2,1)
axes1.hist(xvals,density=True,bins=24)
domain = np.linspace(-4,4,5000)
axes1.plot(domain,norm.pdf(domain))

"""
use normal as g for targe slash
"""

yslash = norm.rvs(size=100000)
denom = []
for y_i in yslash:
    denom.append(slash(y_i)/norm.pdf(y_i))
denom = np.sum(denom)

def w2(x,denom):
    return (slash(x)/norm.pdf(x))/denom

wvals2 = []
for y_i in yslash:
    wvals2.append(np.asscalar(w2(y_i,denom)))

xvals2 = np.random.choice(yslash,size=5000,replace=True,p=wvals2)

axes2 = fig.add_subplot(1,2,2)
axes2.hist(xvals2,density=True,bins=24)
domain = np.linspace(-7,7,5000)
foo = []
for xx in domain:
    foo.append(slash(xx))  
axes2.plot(domain,foo)

"""
Problem 2
"""

x = np.array([8,3,4,3,1,7,2,6,2,7])

def f(_lambda):
    return lognorm.pdf(_lambda,.25,loc=4)

def L(_lambda,x):
    return np.log(np.prod(poisson.pmf(x,_lambda)))

lvals = lognorm.rvs(.25,loc=4,size=1000)
denom = []
for l_i in lvals:
    denom.append(f(l_i)/L(l_i,x))
denom = np.sum(denom)

def w3(_lambda,denom):
    return (f(_lambda)/L(_lambda,x))/denom

wvals3 = []
for l_i in lvals:
    wvals3.append(w3(l_i,denom))
print(sum(wvals3))
    
xvals3 = np.random.choice(lvals,size=5000,replace=True,p=wvals3)
plt.figure()
plt.hist(xvals3,density=True)
domain = np.linspace(0,20,500)
plt.plot(domain,L(domain,x)/L(4.3,x))



