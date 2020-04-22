#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:57:19 2020

@author: claytonfields
"""

##Part c: Implement the methods

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:54:44 2020

@author: claytonfields
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv
import math
import matplotlib.pyplot as plt

def l(alpha,beta,N):
    n = N.size
    temp = []
    retval = np.zeros((alpha.size,1))
    for i in range(n):
        b_i = beta[i].reshape(alpha.size,1)
        N_i = N[i]
        foo = np.asscalar(alpha.T.dot(beta[i]))
        temp.append(N_i*np.log(foo) - np.log(math.factorial(N_i)) - foo)
    for x in temp:
        retval +=x
    return retval
    

def lprime(alpha, beta, N):
    n = N.size
    temp = []
    retval = np.zeros((2,1))
    for i in range(n):
        b_i = beta[i].reshape(2,1)
        N_i = N[i]
        foo = np.asscalar(alpha.T.dot(beta[i]))
        temp.append(N[i]*b_i*(1/foo) - b_i)
    for x in temp:
        retval +=x
    return retval

def l2prime(alpha, beta, N):
    n = N.size
    temp = []
    retval = np.zeros((2,2))
    for i in range(n):
        b_i = beta[i].reshape(2,1)
#        bT_i = beta[i].T.reshape(1,2)
        N_i = np.asscalar(N[i])
        foo = np.asscalar((alpha.T.dot(b_i))**2)
        temp.append(-(N_i/foo)*np.outer(b_i,b_i.T))
    for x in temp:
        retval+=x
    return retval

def fisher(alpha, beta, N):
    n = N.size
    temp = []
    retval = np.zeros((2,2))
    for i in range(n):
        b_i = beta[i].reshape(-1,1)
#        N_i = np.asscalar(N[i])
        foo = np.asscalar(np.dot(alpha.T,b_i))
        temp.append((1/foo)*np.outer(b_i,b_i.T))
    for x in temp:
        retval+=x
    return retval
    
    
data = pd.read_csv('oilspills.dat' ,sep = " ")


b1 = data['importexport'].to_numpy().reshape(26,1)
b2 = data['domestic'].to_numpy().reshape(26,1)
N = data['spills'].to_numpy().reshape(26,1)
beta = np.hstack((b1,b2))
alpha = np.array([0.1,.1]).reshape(2,1)



##Newton's Method
ls = []
t = 0
maxt = 10
newta = [alpha[0]]
newtb = [alpha[1]]
print('Newton\'s method')
for i in range(maxt):
    alpha = alpha - inv(l2prime(alpha, beta, N)).dot(lprime(alpha,beta,N))
    newta.append(alpha[0])
    newtb.append(alpha[1])
    ls.append([t,alpha,inv(l2prime(alpha, beta, N))])
    t+=1
print()
#plt.plot(newta,newtb)

#Print Table
print("Table for MLE using Newton's Method")
df = pd.DataFrame(ls, columns = ['Iteration,t','alpha^\(t)','[M^(t)]^-1'])
df = df.style.hide_index()
df = df.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df

##Fisher Scoring
t = 0
maxt = 10
alpha = np.array([.1,2.5]).reshape(2,1)
fisha = [alpha[0]]
fishb = [alpha[1]]
ls = []

print('Fisher Scoring Method')
for i in range(maxt):   
    alpha = alpha + inv(fisher(alpha, beta, N)).dot(lprime(alpha,beta,N))
    stderr = np.sqrt(np.diag(inv(fisher(alpha, beta, N))))
    fisha.append(alpha[0])
    fishb.append(alpha[1])
    ls.append([t,alpha,inv(fisher(alpha, beta, N))])
    
    t+=1
print()
print('Part D: Find the standard error of the MLE')
print('The standard error of alpha 1 is ',stderr[0])
print('The standard error of alpha 2 is ',stderr[1])
print()

print("Table for MLE using Fisher Information")
df = pd.DataFrame(ls, columns = ['Iteration,t','alpha^\(t)','I^-1'])
df = df.style.hide_index()
df = df.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df

##Ascent
t = 0
maxt = 50
step = 1
alpha = np.array([1,1]).reshape(2,1)
old = alpha
ascenta = [alpha[0]]
ascentb = [alpha[1]]
ls = []
print('Method of Steepest Ascent:')
for i in range(maxt):
    new = old + step*inv(np.eye(2)).dot(lprime(old,beta,N))
    ascenta.append(new[0])
    ascentb.append(new[1])
    if l(new,beta,N)[0] >l(old,beta,N)[0]: #and l(new,beta,N)[1] >l(old,beta,N)[1]:
        ascenta.append(new[0])
        ascentb.append(new[1])
        ls.append([t,new])
        old = new
    else: 
#         ls.append([t,alpha,step])
        step  = step*.5
        
#     ascenta.append(new[0])
#     ascentb.append(new[1])
#     print(new)
    t+=1
print()    
  
print("Table for MLE using Steepest Ascent")
df = pd.DataFrame(ls, columns = ['Iteration,t','alpha^\(t)'])
df = df.style.hide_index()
df = df.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df

##Quasi-Newton Method
t = 0
maxt = 50
alpha = np.array([1,1]).reshape(2,1)
#alpha1 = np.array([.75,75]).reshape(2,1)
M = fisher(alpha,beta,N)
quasia = [alpha[0]]
quasib = [alpha[1]]
ls = []
def updateM(alpha0,alpha1,M):
    z = alpha1 - alpha0
    y = lprime(alpha1,beta,N) - lprime(alpha0,beta,N)
    v = y - M.dot(z)
    c = 1/(v.T.dot(z))
    if np.abs(v.T.dot(z))<.01:
        return M
    if c>0:
        return M
    else:
        
        return  M - c*v.dot(v.T)
    
old = alpha


print('Quasi-Newton\'s Method: without back trace')
for i in range(maxt):
    new = old + inv(M).dot(lprime(old,beta,N))
    quasia.append(new[0]); quasib.append(new[1])
    ls.append([t,new,M])
    M = updateM(old,new,M)
    old=new
    t+=1

print("Table for MLE using Quasi Newton's method: No Backtrack")
df = pd.DataFrame(ls, columns = ['Iteration,t','alpha^\(t)','I^-1'])
df = df.style.hide_index()
df = df.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df


print('Quasi-Newton\'s Method: with back trace')
def pos_def(M):
    if np.array_equal(M, M.T):
        try:
            np.linalg.cholesky(M)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
            return False

def updateMv2(alpha0,alpha1,M):
    z = alpha1 - alpha0
    y = lprime(alpha1,beta,N) - lprime(alpha0,beta,N)
    v = y - M.dot(z)
    c = 1/(v.T.dot(z))
    if np.abs(v.T.dot(z))<.01:
        return M
    if c>0:
        return M
    else:
        M = M - c*v.dot(v.T)
    while(not pos_def(M)):
        c /=2
        M = M - c*v.dot(v.T)
    return M

t = 0
maxt = 50
alpha = np.array([1,1]).reshape(2,1)
old = alpha
print(old)
M = np.eye(2)
for i in range(maxt):
    new = old + inv(M).dot(lprime(old,beta,N))
    quasia.append(new[0]); quasib.append(new[1])
    ls.append([t,new,M])
    M = updateMv2(old,new,M)
    old=new
    print(new)
    t+=1

print("Table for MLE using Quasi Newton's method: with Backtrack")
df = pd.DataFrame(ls, columns = ['Iteration,t','alpha^\(t)','I^-1'])
df = df.style.hide_index()
df = df.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df


#Create contour
x0 = np.linspace(0.01,2.5,100)
y0 = np.linspace(0.01,2.5,100)
X,Y = np.meshgrid(x0,y0)
Zmesh = np.zeros((X.shape[0],Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        a0 = X[i,j]; a1 = Y[i,j]
        a = np.array([a0,a1])
        Zmesh[i,j]=l(a,beta,N)[0]
plt.contourf(X,Y,Zmesh) 
plt.plot(newta,newtb,'ro-')
plt.plot(fisha,fishb,'bo-')
plt.plot(ascenta,ascentb,'go-')
plt.plot(quasia,quasib,'co-')
plt.legend(['Nweton','Fisher','Ascent','Quasi'])
plt.figure()
print()

