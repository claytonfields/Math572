#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:09:46 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
import scipy.stats as stat

"""
Problem 1: From example 2.5
"""
#initialize variables
data = pd.read_csv('facerecognition.dat',sep=" ")
eyediff = data['eyediff'].to_numpy().reshape(1042,1)
n = eyediff.size
y = data['match'].to_numpy().reshape(n,1)
ones = np.ones((1042,1))
Z = np.hstack((ones,eyediff))
#beta = np.array([.9591309,0]).reshape(2,1)
#beta = np.array([0,0]).reshape(2,1)
W = np.zeros((n,n))
b = -np.log(1-pi)
#tol = np.array([.000001,.000001])
k = 0

df = pd.DataFrame({'iter':[], 'beta':[],'hessian':[]})
h = b = {}
betas = [np.array([.95913,0]).reshape(2,1),np.array([0,0]).reshape(2,1)]
b0hista = b1hista = b0histb = b1histb = []
for beta in betas:
    for i in range(10):
        print("iteration: %d" %i, beta)
        pi=(1/(1+np.exp(-1*Z.dot(beta)))).reshape(n,1)
    #    pi = pi.reshape(n,1)
        fill = pi*(1-pi)
        np.fill_diagonal(W,fill)
    #    Hessian = inv(np.dot(Z.T,W).dot(Z))
        Hessian = inv(multi_dot((Z.T,W,Z)))
        h[i]=Hessian
        foo = Z.T.dot(y-pi)
        update = Hessian.dot(foo)
        beta = beta + update
        df['beta'][i] = beta
        b[i] = beta
        k+=1
#        plt.plot(beta[0],beta[1],'ro-')
#    print(k,b[i],h[i])

    

#Create contour
x0 = np.linspace(-5,5,100)
y0 = np.linspace(-15,15,100)
X,Y = np.meshgrid(x0,y0)
def l(y,z,beta1,beta2):
    ones = np.ones((y.shape[0],1))
    beta = np.array([beta1,beta2]).reshape(2,1)
    pi=(1/(1+np.exp(-1*Z.dot(beta)))).reshape(y.size,1)
    b = -np.log(1-pi)
    return multi_dot((y.T,Z,beta))-b.T.dot(ones)
Zmesh = np.zeros((X.shape[0],Y.shape[0]))
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        b0 = X[i,j]; b1 = Y[i,j]
        Zmesh[i,j]=l(y,Z,b0,b1)
plt.contour(X,Y,Zmesh) 
plt.figure()
print()


"""
Problem 2: From problem 2.1 
"""


    
#def cauchy(x,theta,loc):
#    return (1/np.pi)*(theta/((x-loc)**2+theta**2))

def l(x,theta):
    n = x.size
    foo = -np.log([1+(x[i]-theta)**2 for i in range(n)]).sum(axis=0)
#    logsum =  foo.sum(axis=0)
    return  - n*np.log(np.pi) + foo

def lprime(x,theta):
    n = x.size
    foo=[]
#    foo = np.array([2*(x[i]-theta)/((x[i]-theta)**2+1) for i in range (n)]).sum(axis=0)
    for i in range(n):
        foo.append(2*(x[i]-theta)/((x[i]-theta)**2+1))
    return sum(foo)

def l2prime(x,theta):
     n = x.size
     foo=[]
#     num = np.array([-2*(theta**2-2*theta*x[i]+x[i]**2-1) for i in range(n)]).sum(axis=0)
#     denom = np.array([(theta**2-2*theta*x[i]+x[i]**2+1)**2 for i in range(n)]).sum(axis=0)
     for i in range(n):
         foo.append((-2*(theta**2-2*theta*x[i]+x[i]**2-1))/((theta**2-2*theta*x[i]+x[i]**2+1)**2))
     
     return  sum(foo)
 
def h(x,theta):
    return lprime(x,theta)/l2prime(x,theta)
     

domain = np.linspace(-12,12,400)
#plt.plot(domain,cauchy(domain,1,0))

sample = np.asarray([1.77, -.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29,
          3.71, -2.40, 4.53, -.07, -1.05, -13.87, -2.53, -1.75, .27, 43.21])

plt.figure()
plt.plot(domain, l(sample,domain))
avg = sample.mean()
theta0 = [-11,-1,0,1.5,4,4.7,7,8,38,avg]
#plt.plot(theta,l(sample,theta,),'ro')
k=0
np.seterr(all="ignore")
print("Part 2.a: Newton's Method: ")
for theta in theta0:
    string = 'Theta estimate for x0= '+str(theta)+' :'
    while np.abs(h(sample,theta))>.000001:
        k+=1
        if(k>=300):
            print('Failed to Converge')
            break
        theta = theta + h(sample,theta)
    print(string,theta)
    print('Iterations: ',k)
print()



##Bisection


#choose a suitable termination criteria
eps = .000001
kmax = 100

#initialize variable for bisection loop
k =1
xhist = [x]
khist = [k]
ainit= [-2,-1,0,1]
binit = [0,1,2,3]



#perform bisection to find zero of f'(x)
print("Part 2.b: bisection")
for i in range(len(ainit)):
    #set an interval on which to search for max
    a = ainit[i]
    b = binit[i]
    k=0
    #choose x as the middle of the interval
    x = .5*(a+b)
    string= ("Bisection estimate for [%.2f,%.2f]: "%(a,b))
    while ((b-a)>eps):
        k+=1
        if lprime(sample,a)*lprime(sample,x)<=0:
            b=x
        elif lprime(sample,a)*lprime(sample,x)>0:
            a=x
#        ahist.append(a)
#        bhist.append(b)
        x = .5*(a+b)
#        xhist.append(x)
#        k=k+1
#        khist.append(k)
    print(string,x)
    print('Iterations: ',k)
print()
    
##Fixed point interation
alphas = [1,.64,.25]
print('Part 2.c: Fixed-point iteration')
k=0
for alpha in alphas:
    theta = -1 #
    while lprime(sample,theta)>.00000000001:
        theta = theta + alpha*lprime(sample,theta) 
        k+=1    
    print('Fixed-point estimate for alpha = %.2f is: ' %alpha,theta)
    print('Iterations',k)
print()

##Secant Method

#x2 = x1 - fx1*(x1-x0)/(fx1-fx0)
print('Part 2.d: Secant Method')
ainit = [-3,-2,-1,0]
binit = [3,-1,0,2]
for i in range(len(ainit)):
    k=0
    x0=ainit[i]
    x1=binit[i]
    x2=x1+x2
    fx0 = lprime(sample,x0); fx1 = lprime(sample,x1)
    while abs((x0-x1))>.000000000001:
#    for 
        x2 = x1 - lprime(sample,x1)*(x1-x0)/(lprime(sample,x1)-lprime(sample,x0))
    #    fx1 = lprime(sample,x1); fx2 = lprime(sample,x2)
        x0=x1; x1=x2; fx0=fx1; fx1 = lprime(sample,x2)
        k+=1
    print('Secant estimate for [%.2f,%.2f]: '%(ainit[i],binit[i]),x2)
    print('Iterations: ',k)
plt.plot(x2,l(sample,x2,),'g*')


"""
Problem3: From 2.4
"""
a =2
domain= np.linspace(stat.gamma.ppf(0.0, a), stat.gamma.ppf(0.99, a), 100)
probs = stat.gamma.pdf(domain,a)
maxprob = np.max(probs)
xmax = np.argmax(probs)
plt.figure()
plt.plot(domain,probs,lw=5,color='r')
k=0
step = maxprob/10000
#step = .00001
area = 0


while area<=.950000 and k<10000:
    probability = maxprob - k*step
    vals = np.abs(probs-probability)
    l = vals[0:xmax]
    r = vals[xmax:-1]
    lxmin = np.argmin(l)
    rxmin = np.argmin(r)
    lmin = domain[lxmin]
    rmin = domain[rxmin+len(l)]
    area = stat.gamma.cdf(rmin,a) - stat.gamma.cdf(lmin,a)
    k+=1
print('Final are: ',area)
print('Iterations',k)
print('%.4f'%lmin,rmin)

plt.axvline(lmin)
plt.axvline(rmin)
plt.fill_between(lmin,rmin)












    
    
