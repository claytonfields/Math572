#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:33:43 2020

@author: claytonfields
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import expit, logit

#Part A
n = 200

mu1 = 7
sigma1 = 0.5
mu2 = 10
sigma2 = 0.5

delta_true = 0.7

n1 = int(n*delta_true)

y = np.hstack((sigma1*np.random.randn(n1) + mu1, sigma2*np.random.randn(n-n1) + mu2))
print(len(y))

def mix_normal(y, delta):
  return delta*norm.pdf(y, mu1, sigma1) + (1-delta)*norm.pdf(y, mu2, sigma2)

def likelihood(y,delta):
  return np.prod(mix_normal(y, delta))

ax0 = sns.distplot(y, hist=True, kde=False,
                   bins=25, color = 'blue',
                   hist_kws = {'edgecolor':'black'},
                   norm_hist = True,
                   axlabel = 'y',
                   label = 'Histogram of the Data')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax0.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax0.legend(loc='upper right')
ax0.set_title('True PDF and Histogram of Data')
ax0.set_ylabel('Density')



#Part B
m = 10000
deltas = np.zeros(m)
delta = 0.5
for i in range(m):
  # random number generation from the proposal distribution
  delta_star = np.random.rand()
  # compute the Metropolis-Hastings Acceptance Ratio
  MH_ratio = likelihood(y,delta_star)/likelihood(y,delta)
  # accept the delta_star with probability MH_ratio
  r = np.random.rand()
  if r < MH_ratio:
    delta = delta_star
  deltas[i] = delta
  
fig, axes = plt.subplots(1, 2)

ax_unif = sns.distplot(deltas, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = r'$\delta$',
             label = 'Histogram of the samples',
             ax=axes[0])

ax_unif.set_title('KDE and Histogram of $\delta$ Samples with UNIF(0,1) Prior')
ax_unif.set_ylabel('Density')

axes[1].plot(deltas)
  
#Part C: Random Walk Chain
m = 10000
deltas2 = np.zeros(m)
delta = .5


for i in range(m):
    #random number generation from proposal dist
    delta_star = delta + np.random.uniform(-1,1)
    
     #comupte the Metropolis-Hastings Algorithm
    if delta_star < 0 or delta_star > 1:
        MH_ratio = 0
    else : 
        MH_ratio = likelihood(y,delta_star)/likelihood(y,delta)
    #Accet the delt_star iwth propability MH_ration
    r = np.random.rand()
    if r< MH_ratio:
        delta = delta_star
    deltas2[i] = delta 
  
plt.figure()
sns.distplot(deltas2, hist=True, kde=True,bins=25).set_title('Random Walk Chain Hist.')
#fig, axs = plt.subplots(1, 2, constrained_layout=True)
#sns.distplot(deltas2, hist=True, kde=True,bins=25,ax=axs[0, 0]).set_title('Random Walk Chain Hist.')
#axs[0,1].plot(ax,deltas2)
#axs[0,1].set_title('Random Walk Trace')
plt.figure()
plt.plot(deltas2)
plt.title('Random Walk Trace')

# part c: U-transform Random Walk Chain
def mix_normal(y,u):
    return expit(u)*norm.pdf(y, mu1,sigma1)+(1-expit(u))*norm.pdf(y,mu2,sigma2)

def likelihood(u):
    return np.prod(mix_normal(y,u))


m = 10000
us = np.zeros(m)
delta = .5
u = logit(delta)
for i in range(m):
    #random number generation from proposal dist
    
    
    u_star = u + np.random.uniform(-.1,.1)
    #print(u_star)
    #if delta_star < 0:
    #    delta_star = 0
    J_u = np.abs(1/(np.exp(u)+1)-1/(np.exp(u)+1)**2)
    J_u_star = np.abs(1/(np.exp(u_star)+1)-1/(np.exp(u_star)+1)**2)  
    #J_del = -1/(delta*(delta-1))
    #J_del = -1/(delta_star*(delta-1))
    #print(J_u)                                             
    #comupte the Metropolis-Hastings Algorithm                                       
    MH_ratio = (likelihood(u_star)*J_u_star)/(likelihood(u)*J_u)
    #Accet the delt_star iwth propability MH_ration
    r = np.random.rand()
    if r< MH_ratio:
        u = u_star
       
    us[i] = u

plt.figure()
sns.distplot(expit(us), hist=True, kde=True,bins=25).set_title('U-Space Transform Chain Hist.') 
plt.figure()
plt.plot(expit(us))
plt.title('U-space Transform Trace')



#fig, axs = plt.subplots(3, 2, constrained_layout=True)
#sns.distplot(deltas, hist=True, kde=True,bins=25,ax=axs[0, 0]).set_title('Independence Chain Hist.')
#axs[0,1].plot(ax,deltas)
#axs[0,1].set_title('Independence Trace')
#sns.distplot(deltas2, hist=True, kde=True,bins=25,ax=axs[1, 0]).set_title('Random Walk Chain Hist.')
#axs[1,1].plot(ax,deltas2)
#axs[1,1].set_title('Random Walk Trace')
#sns.distplot(expit(us), hist=True, kde=True,bins=25,ax=axs[2, 0]).set_title('U-Space Transform Chain Hist.')
#axs[2,1].plot(ax,expit(us))
#axs[2,1].set_title('U-space Transform Trace')

#Problem 2: From 7.2 #Not working
#def mix_normal(y, delta):
#  return delta*norm.pdf(y, mu1, sigma1) + (1-delta)*norm.pdf(y, mu2, sigma2)
#
#def likelihood(y,delta):
#  return np.prod(mix_normal(y, delta))
#
#m = 10000
#deltas2a = np.zeros(m)
#delta = 0.5
#sigma = .1
#mu=0
#for i in range(m):
#  # random number generation from the proposal distribution
#  delta_star = sigma*np.random.randn()+mu
#  # compute the Metropolis-Hastings Acceptance Ratio
#  MH_ratio = likelihood(y,delta_star)/likelihood(y,delta)
#  # accept the delta_star with probability MH_ratio
#  r = np.random.rand()
#  if r < MH_ratio:
#    delta = delta_star
#  deltas2a[i] = delta
#  
#fig, axes = plt.subplots(1, 2)
#
#ax_unif = sns.distplot(deltas2a, hist=True, kde=True,
#             bins=25, color = 'blue',
#             hist_kws={'edgecolor': 'black'},
#             axlabel = r'$\delta$',
#             label = 'Histogram of the samples',
#             ax=axes[0])
#
#ax_unif.set_title('KDE and Histogram of $\delta$ Samples with UNIF(0,1) Prior')
#ax_unif.set_ylabel('Density')
#
#axes[1].plot(deltas2a)
  


