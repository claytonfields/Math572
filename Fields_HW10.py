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
np.random.seed(88)
mu1 = 7
sigma1 = 0.5
mu2 = 10
sigma2 = 0.5
delta_true = 0.7
n1 = int(n*delta_true)
#Create Random Sample
y = np.hstack((sigma1*np.random.randn(n1) + mu1, sigma2*np.random.randn(n-n1) + mu2))

def mix_normal(y, delta):
  return delta*norm.pdf(y, mu1, sigma1) + (1-delta)*norm.pdf(y, mu2, sigma2)

def likelihood(y,delta):
  return np.prod(mix_normal(y, delta))

ax2a = sns.distplot(y, hist=True, kde=False,
                   bins=25, color = 'blue',
                   hist_kws = {'edgecolor':'black'},
                   norm_hist = True,
                   axlabel = 'y',
                   label = 'Histogram of the Data')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax2a.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax2a.legend(loc='upper right')
ax2a.set_title('True PDF and Histogram of Data')
ax2a.set_ylabel('Density')


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
ax_2b = sns.distplot(deltas[200:], hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = r'$\delta$',
             label = 'Data',
             ax=axes[0])
ax_2b.set_title('Density: Independence Chain')
ax_2b.legend()
axes[1].plot(deltas)
axes[1].set_title('Independence Trace')
  
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
  
fig, axes = plt.subplots(1, 2)
ax_2c = sns.distplot(deltas2[200:], hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_2c.set_title('Density: Random Walk Chain')
ax_2c.legend()
axes[1].plot(deltas2[200:])
axes[1].set_title('Random Walk Trace')


#Part D: U-transform Random Walk Chain
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
    u_star = u + np.random.uniform(-1,1)
    #Jacobian of expit
    J_u = np.abs((np.exp(u))/(np.exp(u)+1)**2)
    J_u_star = np.abs((np.exp(u_star))/(np.exp(u_star)+1)**2)                 
    #comupte the Metropolis-Hastings Algorithm                                       
    MH_ratio = (likelihood(u_star)*J_u_star)/(likelihood(u)*J_u)
    #Accet the delt_star iwth propability MH_ration
    r = np.random.rand()
    if r< MH_ratio:
        u = u_star   
    us[i] = u

fig, axes = plt.subplots(1, 2)
ax_2d = sns.distplot(expit(us[200:]), hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_2d.set_title('Density: U-space Transform')
ax_2d.legend()
axes[1].plot(expit(us))
axes[1].set_title('U Transform Trace')


#Problem 2: From 7.2 
#PartA
np.random.seed(22)
m = 10000
vals_0 = np.zeros(m)
vals_7 = np.zeros(m)
vals_15 = np.zeros(m)
x_0 = 0
x_7 = 7
x_15 = 15
sigma = 2.5
mu=7
delta = .7

def mix_normal(y, delta):
  return delta*norm.pdf(y, mu1, sigma1) + (1-delta)*norm.pdf(y, mu2, sigma2)

for i in range(m):
  # random number generation from the proposal distribution
  xstar_0 = norm.rvs(size=1,loc=x_0,scale=.01)
  xstar_7 = norm.rvs(size=1,loc=x_7,scale=.01)
  xstar_15 = norm.rvs(size=1,loc=x_15,scale=.01)
  # compute the Metropolis-Hastings Acceptance Ratio
  R_0 = (mix_normal(xstar_0,delta)*norm.rvs(loc=x_0,scale=sigma,size=1))/(mix_normal(x_0,delta)*norm.rvs(loc=xstar_0,scale=sigma,size=1))
  R_7 = (mix_normal(xstar_7,delta)*norm.rvs(loc=x_7,scale=sigma,size=1))/(mix_normal(x_7,delta)*norm.rvs(loc=xstar_7,scale=sigma,size=1))
  R_15 = (mix_normal(xstar_15,delta)*norm.rvs(loc=x_15,scale=sigma,size=1))/(mix_normal(x_15,delta)*norm.rvs(loc=xstar_15,scale=sigma,size=1))
  #Accept the delta_star iwth propability MH_ratio
  r = np.random.rand()
  if r < min(R_0,1):
    x_0=xstar_0
  if r < min(R_7,1):
    x_7=xstar_7
  if r < min(R_15,1):
    x_15=xstar_15
  vals_0[i] = x_0
  vals_7[i] = x_7
  vals_15[i] = x_15
  
fig, axes = plt.subplots(1, 2)
ax_3a0 = sns.distplot(vals_0, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_3a0.set_title('Density: 7.a x^(0)=0')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3a0.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3a0.legend()
axes[1].plot(vals_0)
axes[1].set_title('Trace for x^(0)=0')

fig, axes = plt.subplots(1, 2)
ax_3a7 = sns.distplot(vals_7, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_3a7.set_title('Density: 7.a x^(0)=7')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3a7.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3a7.legend()
axes[1].plot(vals_7)
axes[1].set_title('Trace for x^(0)=7')

fig, axes = plt.subplots(1, 2)
ax_3a15 = sns.distplot(vals_15, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_3a15.set_title('Density: 7.a x^(0)=15')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3a15.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3a15.legend()
axes[1].plot(vals_15)
axes[1].set_title('Trace for x^(0)=15')

#PartB
m = 10000
vals_0 = np.zeros(m)
vals_7 = np.zeros(m)
vals_15 = np.zeros(m)
x_0 = 0
x_7 = 7
x_15 = 15
sigma = 2.5
sigma_p = 2
mu=7
delta = .7

def mix_normal(y, delta):
  return delta*norm.pdf(y, mu1, sigma1) + (1-delta)*norm.pdf(y, mu2, sigma2)


for i in range(m):
  # random number generation from the proposal distribution
  xstar_0 = norm.rvs(size=1,loc=x_0,scale=sigma_p)
  xstar_7 = norm.rvs(size=1,loc=x_7,scale=sigma_p)
  xstar_15 = norm.rvs(size=1,loc=x_15,scale=sigma_p)
  # compute the Metropolis-Hastings Acceptance Ratio
  R_0 = (mix_normal(xstar_0,delta)*norm.rvs(loc=x_0,scale=sigma,size=1))/(mix_normal(x_0,delta)*norm.rvs(loc=xstar_0,scale=sigma,size=1))
  R_7 = (mix_normal(xstar_7,delta)*norm.rvs(loc=x_7,scale=sigma,size=1))/(mix_normal(x_7,delta)*norm.rvs(loc=xstar_7,scale=sigma,size=1))
  R_15 = (mix_normal(xstar_15,delta)*norm.rvs(loc=x_15,scale=sigma,size=1))/(mix_normal(x_15,delta)*norm.rvs(loc=xstar_15,scale=sigma,size=1))
  #Accept the delta_star iwth propability MH_ratio
  r = np.random.rand()
  if r < min(R_0,1):
    x_0=xstar_0
  if r < min(R_7,1):
    x_7=xstar_7
  if r < min(R_15,1):
    x_15=xstar_15
  vals_0[i] = x_0
  vals_7[i] = x_7
  vals_15[i] = x_15
  
fig, axes = plt.subplots(1, 2)
ax_3b0 = sns.distplot(vals_0, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_3b0.set_title('Density: 7.b x^(0)=0')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3b0.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3b0.legend()
axes[1].plot(vals_0)
axes[1].set_title('Trace for x^(0)=0')

fig, axes = plt.subplots(1, 2)
ax_3b7 = sns.distplot(vals_7, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ax_3b7.legend()

ax_3b7.set_title('Density: 7.b x^(0)=7')
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3b7.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3b7.legend()
axes[1].plot(vals_7)
axes[1].set_title('Trace for x^(0)=7')

fig, axes = plt.subplots(1, 2)
ax_3b15 = sns.distplot(vals_15, hist=True, kde=True,
             bins=25, color = 'blue',
             hist_kws={'edgecolor': 'black'},
             axlabel = '',
             label = 'Data',
             ax=axes[0])
ygrid = np.linspace(mu1-3*sigma1, mu2+3*sigma2, 100)
ax_3b15.plot(ygrid, mix_normal(ygrid, delta_true), color='Green', label='True PDF')
ax_3b15.set_title('Density: 7.b x^(0)=15')
ax_3b15.legend()
axes[1].plot(vals_15)
axes[1].set_title('Trace for x^(0)=15')