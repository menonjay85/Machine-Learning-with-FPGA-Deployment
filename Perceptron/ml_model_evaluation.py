# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:42:47 2024

@author: menon
"""

import pandas as pd
import numpy as np

data = [
    [-2,4,-1,-1],
    [4,1,-1,-1],
    [1,6,-1,1],
    [2,4,-1,1],
    [6,2,-1,1]]

df = pd.DataFrame(data, columns=['x0', 'x1', 'x2', 'y'])
df.index = df.index + 1
print(df)

X = df.iloc[:,0:4].values
y = df['y'].values
Size = len(y)
I = np.ones(Size,float)
A = np.column_stack((I,X))

w_init = np.random.uniform(-1, 1, A.shape[1])
alpha = 0.001
epsilon = 1e-5

while True:
    w_update = np.zeros(w_init.shape)
    
    for i in range(Size):
        z = np.dot(w_init, A[i])
        # print('Total loss sum: ', z)
        error = -y[i]*z
        # print('Error: ', error)
        
        if error > 0:
            w_update += alpha * y[i] * A[i]
    
    w_init += w_update
    
    if np.linalg.norm(w_update) < epsilon:
        break
    
print("Final Weights: ", w_init)


###
## Second part  ##
###

from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# `y_pred` is the predicted value of `y`
# `y_actual` is the actual value of `y`
# `iteration` is the iteration number
def plot_error(y_pred, y_actual, iteration):
    error = y_pred - y_actual
    plt.plot(iteration, error, 'ro')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error between predicted and actual values')
    plt.show()

df_d1 = pd.read_csv('Dataset_1.csv')
print(df_d1)
df_d1.index = df_d1.index + 1
X_d1 = df_d1.iloc[:,0:4].values
y_d1 = df_d1['y'].values
Size_d1 = len(y_d1)
I_d1 = np.ones(Size_d1,float)
A_d1 = np.column_stack((I_d1,X_d1))
# print(A_d1)

w_init_d1 = np.random.uniform(-1, 1, A_d1.shape[1])
alpha_d1 = 0.001
epsilon_d1 = 1e-5
X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(A_d1, y_d1, test_size=0.2, random_state=42)

# print("Training set size:", len(X_train_d1))
# print("Testing set size:", len(X_test_d1))

while True:
    w_update_d1 = np.zeros(w_init_d1.shape)
    
    for i in range(len(X_train_d1)):
        z_d1 = np.dot(w_init_d1, X_train_d1[i])
        # print('Total loss sum: ', z_d1)
        error_d1 = -y_train_d1[i]*z_d1
        # print('Error: ', error_d1)
        
        if error_d1 > 0:
            w_update_d1 += alpha_d1 * y_train_d1[i] * X_train_d1[i]
    
    w_init_d1 += w_update_d1
    print("Weights in Iteration ", i, ": ", w_init_d1)
    print('Error: ', error_d1)
    
    y_pred_d1 = np.sign(np.dot(X_test_d1, w_init_d1))
    print("Predictions in iteration ", i, ": ", y_pred_d1)

    if np.linalg.norm(w_update_d1) < epsilon_d1:
        break
    
print("Final Weights for Dataset 1: ", w_init_d1)

y_pred_d1 = np.sign(np.dot(X_test_d1, w_init_d1))
print("Predictions: ", y_pred_d1)

accuracy_d1 = accuracy_score(y_test_d1, y_pred_d1)
print(accuracy_d1)

results_d1 = pd.DataFrame({'Actual': y_test_d1, 'Predicted': y_pred_d1})
print(results_d1)


print('Accuracy: %.2f' % accuracy_score(y_test_d1, y_pred_d1))