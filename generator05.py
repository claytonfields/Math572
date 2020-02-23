#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:53:21 2020

@author: claytonfields
"""

import sympy as sp

symbols = sp.symbols
r, t, K, N, N0 = symbols('r t K N N0')

LSE = (sp.log(N)-sp.log((K*N0)/(N0 + (K-N0)*sp.exp(-r*t))))**2
dLdr = sp.diff(LSE,r,1)
dldK = sp.diff(LSE,K,1)