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

alp = 1e-3
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


def grad_descent(func, winital, alp, total_iterations, tol):
    begin_time = process_time()
    
    i = 1
    w_update_mag = 1.
    absDg = 1
    errors = []
    
    while ((i < total_iterations) and (w_update_mag > tol)):
        grad_func = grad(func)
        Dg = grad_func(winital)
        absDg = np.linalg.norm(Dg)
        winital = winital - (alp)*Dg
        w_update_mag = (absDg)*(alp)
        errors.append(w_update_mag)
        i+=1
        
    end_time = process_time()
    delta_t = (end_time - begin_time) * 1000
    
    # print(errors[1:10])
    # print(errors[-10:-1])
    
    # print("Total iterations: " , i)
    # print("Total time %.3g" %delta_t, " ms")
    # print("Time per iteration: ", (delta_t/i))
    # print("Learning rate: ", alp, "\n")
    # plot_errors(errors)
    
    return (winital, i, w_update_mag, delta_t, errors)



def grad_descent_new(func, winital, alp, total_iterations, tol):
    begin_time = process_time()
    
    i = 1
    w_update_mag = 1.
    absDg = 1
    errors = []
    
    while ((i < total_iterations) and (w_update_mag > tol)):
        alp_new = alp/i
        grad_func = grad(func)
        Dg = grad_func(winital)
        absDg = np.linalg.norm(Dg)
        winital = winital - (alp_new)*Dg
        w_update_mag = (absDg)*(alp_new)
        errors.append(w_update_mag)
        i+=1
        
    end_time = process_time()
    delta_t = (end_time - begin_time) * 1000
    
    # print(errors[1:10])
    # print(errors[-10:-1])
    
    # print("Total iterations: " , i)
    # print("Total time %.3g" %delta_t, " ms")
    # print("Time per iteration: ", (delta_t/i))
    # print("Learning rate: ", alp_new, "\n")
    # plot_errors(errors)
    
    return (winital, i, w_update_mag, delta_t, errors)


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
    
    # print(errors[1:10])
    # print(errors[-10:-1])
    
    # print("Total iterations: " , i)
    # print("Total time %.3g" %delta_t, " ms")
    # print("Time per iteration: ", (delta_t/i))
    # print("Learning rate: ", alpha, "\n")
    # plot_errors(errors)
    
    return (winital, i, w_update_mag, delta_t, errors)




alp_vals = [0.1, 0.01, 0.001]


ModelResults = []

for alp in alp_vals:

    w_final1, count1, w_update1, delta_t1, errors1 = grad_descent(f, w_init, alp, tot_iter, tol)
    w_final2, count2, w_update2, delta_t2, errors2 = grad_descent_new(f, w_init, alp, tot_iter, tol)
    w_final3, count3, w_update3, delta_t3, errors3 = grad_descent_normalized(f, w_init, alp, tot_iter, tol)
    
    # Compile results
    ModelResults.append({
        'method': 'Constant Learning Rate',
        'alp': alp,
        'final_w': w_final1,
        'iterations': count1,
        'final_update': w_update1,
        'total_time_ms': delta_t1,
        # 'errors': errors1
    })
    
    ModelResults.append({
        'method': 'Learing rate decreasing with iterations',
        'alp': alp,
        'final_w': w_final2,
        'iterations': count2,
        'final_update': w_update2,
        'total_time_ms': delta_t2,
        # 'errors': errors2
    })
    
    ModelResults.append({
        'method': 'Normalized method',
        'alp': alp,
        'final_w': w_final3,
        'iterations': count3,
        'final_update': w_update3,
        'total_time_ms': delta_t3,
        # 'errors': errors3
    })

# Output the results for comparison
for result in ModelResults:
    print(result)

plot_errors(errors1)
plot_errors(errors2)
plot_errors(errors3)
