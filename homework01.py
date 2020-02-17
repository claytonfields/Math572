#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:51:46 2020

@author: claytonfields
"""
from math import log
import matplotlib.pyplot as plt
import numpy as np


#define function f(x) to be optimized
def f(x):
    return np.log(x)/(1+x)

#define the derivative of f(x)
def fprime(x):
    return (1+1/x-np.log(x))/(1+x)**2

#set an interval on which to search for max
a = 1.0
b = 5.0

#choose x as the middle of the interval
x = .5*(a+b)

#choose a suitable termination criteria
eps = .000001
kmax = 100

#initialize variable for bisection loop
k =1
xhist = [x]
khist = [k]
ahist = [a]
bhist = [b]

#perform bisection to find zero of f'(x)
print('k       x_k')
while ((b-a)>eps):
    if fprime(a)*fprime(x)<=0:
        b=x
    elif fprime(a)*fprime(x)>0:
        a=x
    ahist.append(a)
    bhist.append(b)
    x = .5*(a+b)
    xhist.append(x)
    k=k+1
    khist.append(k)
    print(k, '      ', x)
print()
print()
print('The value of x where the max occurs is ',x)    
print()
print('The maximum value of f(x) is ',f(x))
print()
print('The value of f\'(x) is ',fprime(x))

#domanins for plotting
xx = np.linspace(1,5,100)
xk = np.linspace(1,k,k)

#plot the results of bisection
plt.plot(xx,f(xx))
plt.title('f(x) with Maximum Value')
plt.plot(x,f(x))
plt.axvline(x=x, color='black', label="x=3.59112")
plt.plot(x,f(x),'ro')

plt.figure()
plt.title('fprime(x) with Descending Bisection Intervals')
plt.plot(xx,fprime(xx))
plt.plot(x,fprime(x),'ro')
for i in range(len(ahist)):
#    plt.axhline()
    y = -.1+-.05*i
    plt.plot(ahist[i],y,'r')
    plt.plot(bhist[i],y,'b')
    plt.hlines(y=y,xmin=ahist[i],xmax=bhist[i],label=str(i+1))
plt.ylim(-.6,.65)

plt.figure()
plt.plot(khist,xhist)
plt.title('x against k')