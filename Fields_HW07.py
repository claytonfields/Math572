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

np.seterr(all="ignore")



accepted = []
k=0
while len(accepted)<5000:
    Z = np.random.randn()
    U = np.random.rand()
    
    if(U <= qOverE(Z) and t(Z)>0):
        accepted.append(t(Z))
    k+=1
    



print("The acceptance ratio for problem 1 is: ",5000/k)
##Problem 01 Plot01
xx = np.linspace(-3,3,100)
plt.plot(xx,q(xx),'r--',linewidth=2)
plt.plot(xx,e(xx))
plt.legend(["target","envelope"])
plt.title("Problem 1: Unnormalized Density")
##Problem 01 Plot02
plt.figure()
plt.hist(accepted, density=True)
xx = np.linspace(0,12,100)
plt.plot(xx,st.gamma.pdf(xx,r))
sns.distplot(accepted,kde=True,hist=False,label="data")
plt.legend(["gamma","kde"])
plt.title("Problem 1: Historgram and Density Curves")



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


##Problem 02 Plot01

plt.figure()
xx = np.linspace(0,20,200)
plt.plot(xx,q2(xx))
plt.plot(xx,1.5*st.gamma.pdf(xx,2,scale=2))

plt.legend(["target","envelope"])
plt.title("Problem 2: Unnormalized Density")

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


print("The acceptance ratio for problem 2 is: ",5000/k)
##Problem 02 Plot02
domain = np.linspace(0,20,500)
plt.hist(accepted2,density=True)
sns.distplot(accepted2,kde=True,hist=False,label="data")
plt.plot(domain,q2(domain))
plt.title("Problem 2: Historgram and Density Curves")
plt.legend(["kde","true pdf"])
print(5000/k)
domain = np.linspace(0,20,500)
plt.hist(accepted2,density=True)
sns.distplot(accepted2,kde=True,hist=False)
plt.plot(domain,q2(domain))

