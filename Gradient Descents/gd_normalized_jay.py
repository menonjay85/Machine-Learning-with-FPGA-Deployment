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

alpha = 1e-3
tot_iter = 3000
tol = 1e-6

def f(w):
    return (a + np.dot(b.T, w) + np.dot(w.T, np.dot(C, w)))



def plot_errors(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def grad_descent_normalized(func, winital, alpha, total_iterations, tol):
    begin_time = process_time()
    
    i = 1
    w_update_mag = 1.
    absDg = 1
    errors = []
    
    while ((i < total_iterations) and (w_update_mag > tol)):
        
        grad_func = grad(func)
        Dg = grad_func(winital)
        absDg = np.linalg.norm(Dg)
        norm_Dg = Dg / absDg
        if absDg < 1e-8:
            break
        winital = winital - (alpha)*norm_Dg
        w_update_mag = alpha * absDg
        errors.append(w_update_mag)
        i+=1
        
    end_time = process_time()
    delta_t = (end_time - begin_time) * 1000
    
    print(errors[1:10])
    print(errors[-10:-1])
    
    print("Total iterations: " , i)
    print("Total time %.3g" %delta_t, " ms")
    print("Time per iteration: ", (delta_t/i))
    print("Learning rate: ", alpha, "\n")
    plot_errors(errors)
    
    return (winital, i, w_update_mag, delta_t, errors)



grad_descent_normalized(f, w_init, alpha, tot_iter, tol)
