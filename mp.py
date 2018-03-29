#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:38:30 2018
@author: wsj

"""
import numpy as np
import random
import matplotlib.pyplot as plt
def mp(y,phi,psi,K):
    A = phi*psi.H
    M,N = np.shape(A)
    s = np.array(np.zeros((N,1)))
    v = y
    num_iters = 4*K
    for i in range(num_iters):
        product = A.H*v
        g = (abs(product)).tolist()
        pos = g.index(max(g))
        scale_product = product[pos]/(np.dot(A[:,pos].H,A[:,pos]))
        s[pos] = s[pos] + scale_product
        v = v - A[:,pos]*scale_product
        if np.linalg.norm(v)<1e-6:
            break
    return psi.H*s
def my_plot(x,x_r):
    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(8,6))  
    line1, = axes.plot(x_r,'r-s')
    line2, = axes.plot(x,'b--o')
    axes.legend((line1,line2),('recovered signal','src signal'),loc = 'upper right')
    axes.set_xlabel(u'time')
    axes.set_ylabel(u'signal')
    axes.set_title('mp algorithm')
if __name__ == '__main__':
    N = 256
    M = 64
    K = 8
    x = np.zeros((N,1))
    index_k = random.sample(range(N),K)
    x[index_k] = 5*np.random.randn(K,1)
    psi = np.eye(N,N);
    phi = np.random.randn(M,N);
    phi = np.mat(phi)
    psi = np.mat(psi)
    x = np.mat(x)
    y = phi*psi.H*x
    my_plot(mp(y,phi,psi,K),x)
#    print(mp(y,phi,psi,K)-x)
    