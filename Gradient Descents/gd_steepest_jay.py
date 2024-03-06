# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:56:47 2024

@author: menon
"""

from autograd import grad
from autograd import numpy as np
from time import process_time
import matplotlib.pyplot as plt


a = 20
b = np. array([1. , -1.])
b= b.reshape(2,1)
C = np.array([[1.,0.],[0.,2.]])

w_init = np.array([-22, 33]).reshape(2,1)

alpha = 1e-2
tot_iter = 3000
tol = 1e-6

z = (1+2.236)/2 ## Golden ratio

def f(w):
    return (a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w)))

def goldendelta(x4,x1,z):
    return((x4-x1)/z)

def plot_errors(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def goldensearch(g,w,h,x1,x4,accuracy):
    ## initial positions of the four points
    ## x1 = sigma/10
    ## x4 = sigma*10
    x2 = x4 - goldendelta(x4,x1,z)
    x3 = x1 + goldendelta(x4,x1,z)
    ## initial values of the function at the four points
    f1 = g(w-x1*h); f2 = g(w-x2*h); f3 = g(w-x3*h); f4 = g(w-x4*h);
    i = 0
    error = abs(x4-x1)
    while error > accuracy:
        if (f2<f3): 
            x4,f4 = x3,f3
            x3,f3 = x2,f2
            x2 = x4 - goldendelta(x4,x1,z)
            f2 = g(w-x2*h)
        else:
            x1,f1 = x2,f2
            x2,f2 = x3,f3
            x3 = x1 + goldendelta(x4,x1,z)
            f3 = g(w-x3*h)
        i += 1
        error = abs(f4-f1)
    return((x1+x4)/2.0,i,error)



def golden(g,w,h,alpha):
    alpha,iter,error = goldensearch(g,w,h,alpha/10.,alpha*10.0,1e-6)
    return alpha


def grad_descent_steep(func, winital, alpha, total_iterations, tol):
    begin_time = process_time()
    
    w = winital.copy()
    i = 1
    w_update_mag = 1.
    absDg = 1
    errors = []
    
    for i in range(total_iterations):
        grad_func = grad(func)
        Dg = grad_func(winital)
        absDg = np.linalg.norm(Dg)
        if absDg < 1e-8:
            break
        
        optim_alpha = golden(lambda alpha: func(w - alpha * Dg), w, Dg, alpha)
        winital = winital - (optim_alpha)*Dg
        
        w_update_mag = optim_alpha * absDg
        errors.append(w_update_mag)
        i+=1
        
    end_time = process_time()
    delta_t = (end_time - begin_time) * 1000
    
    print(errors[1:10])
    print(errors[-10:-1])
    
    print("Total iterations: " , i)
    print("Total time %.3g" %delta_t, " ms")
    print("Time per iteration: ", (delta_t/i))
    print("Learning rate: ", optim_alpha, "\n")
    plot_errors(errors)
    
    return (winital, i, w_update_mag, delta_t, errors)



grad_descent_steep(f, w_init, alpha, tot_iter, tol)
