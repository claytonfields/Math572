# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

r = 2
a = r-1/3
b = 1/np.sqrt(9*a)

def t(z):
    return a*(1+b*z)**3


def qOverE(Z):
    return np.exp(Z**2/2 + a*np.log(t(Z)/a)- t(Z)+a)

def q(z):
    return np.exp(a*np.log(t(z)/a)-t(z)+a)

def e(z):
    return np.exp(-z**2/2)

accepted = []
k=0
while len(accepted)<5000:
    Z = np.random.randn()
    U = np.random.rand()
    
    if(U <= qOverE(Z) and t(Z)>0):
        accepted.append(t(Z))
    k+=1
    


print(5000/k)
plt.hist(accepted, density=True)
xx = np.linspace(0,12,100)
plt.plot(xx,st.gamma.pdf(xx,r))
sns.distplot(accepted,kde=True,hist=False)
plt.figure()
xx = np.linspace(-3,3,100)
plt.plot(xx,q(xx),'r--',linewidth=2)
plt.plot(xx,e(xx))
#xx = np.linspace(0,12,5000)
#sns.kdeplot(xx,accepted)


"""
Problem 02
"""

def q2(x):
    return (1/12)*(1+x)*np.exp(-(x-1)**2/(2*x))

def e2(x):
    return 1.5*st.gamma.pdf(x,2,scale=2)

plt.figure()
xx = np.linspace(0,20,200)
plt.plot(xx,q2(xx))
plt.plot(xx,1.5*st.gamma.pdf(xx,2,scale=2))
plt.figure()
#zz = f(xx)- 1.5*st.gamma.pdf(xx,2,scale=2)
accepted2 = []
k=0
while len(accepted2)<5000:
    Y = np.random.gamma(2,scale=2.1)
    U = np.random.rand()
    if(U<=q2(Y)/e2(Y)):
        accepted2.append(Y)
    k+=1

print(5000/k)
domain = np.linspace(0,20,500)
plt.hist(accepted2,density=True)
sns.distplot(accepted2,kde=True,hist=False)
plt.plot(domain,q2(domain))