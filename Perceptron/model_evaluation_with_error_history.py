# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:57:05 2024

@author: menon
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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
# print(df)

X = df.iloc[:,0:4].values
y = df['y'].values
Size = len(y)
I = np.ones(Size,float)
A = np.column_stack((I,X))

w_init = np.random.uniform(-1, 1, A.shape[1])
alpha = 0.001
epsilon = 1e-5
X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=42)

error_history = []

while True:
    w_update = np.zeros(w_init.shape)
    errors1 = []
    
    for i in range(len(X_train)):
        z = np.dot(w_init, X_train[i])
        # print('Total loss sum: ', z_d1)
        error = -y_train[i]*z
        # print('Error: ', error_d1)
        
        if error > 0:
            w_update += alpha * y_train[i] * X_train[i]
        
        errors1.append(abs(error))
    
    error_history.append(np.mean(errors1))
    w_init += w_update
    y_pred = np.sign(np.dot(X_test, w_init))
    print("Predictions in iteration ", i, ": ", y_pred)
    print("Weights in Iteration ", i, ": ", w_init)
    print('Error: ', error)

    if np.linalg.norm(w_update) < epsilon:
        break



plt.plot(range(1, len(error_history) + 1), error_history)
plt.show()

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

miss_rate = 1 - accuracy
print("Number of misclassified cases: ", miss_rate*len(y_pred))