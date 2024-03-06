# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:44:40 2024

@author: menon
"""

###
### Question 4
###

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


df = pd.read_csv('Dataset_1.csv')
cols = df.columns
ftrs = 4

X = df.iloc[:,0:ftrs].values
y = df['y'].values

std_scl = StandardScaler()
X_std = std_scl.fit_transform(X[:,0:ftrs])
ones = np.ones(len(X_std[:,0]),dtype=float)
X_std_ = np.column_stack((ones,X_std))

X_train, X_test, y_train, y_test = train_test_split(X_std_,y,test_size=0.33,random_state=0)


def act(Z):
    if (Z >= 0.0): 
        return(1)
    else: 
        return(-1)

def gd(A,W,y):
    return (2*((A*W)-y)*A)/1

def model(A,w):
    return np.matmul(A,w)

def sgd(X, w, Y, eta):
    epochs = 100
    errors = []
    t_e = 1
    tol = 1e-5
    Nobs = 3
    t = 0
    A = X
    while (t<=epochs and t_e/Nobs > tol):
        t_e = 0.0
        for i, x in enumerate(A):
            wlast=w
            dw = gd(A[i],w,Y[i])
            w = w - eta*dw
            error = np.dot((wlast - w),(wlast - w))
            t_e += error            
        
        errors.append(t_e)
        t += 1
    return w,errors,t      

def final(X,w,y):
    y_pred = []
    sum_error = 0.0
    for i, x in enumerate(X):
        yp = act(model(x,w))
        error = y[i]-yp
        sum_error += error*error
        y_pred.append(yp)
    print('total error:', sum_error)
    print('accuracy: %.2f' % accuracy_score(y, y_pred))
    print('number of misclassified: %d' % (y != y_pred).sum())
    return y_pred,sum_error

w = np.ones(len(X_train[0]))*0.1          
w,errors,t = sgd(X_train,w,y_train,0.0001)
print('epochs: ', t-1, '\nfinal weights= ', w)

plt.plot(errors)
plt.xlabel('target')
plt.ylabel('prediction')
plt.grid()
plt.show()

print('\ntraining: ')
y_pred_train,sum_error = final(X_train,w,y_train)
print('\ntest: ')
y_pred_test,sum_error = final(X_test,w,y_test)


###
### Question 5
###

w_custom, cust_error, t_custom = sgd(X_train, np.ones(len(X_train[0])) * 0.1, y_train, 0.0001)

# Create and train the sklearn Perceptron model
perceptron_sklearn = Perceptron(max_iter=100, eta0=0.0001, random_state=0)
perceptron_sklearn.fit(X_train, y_train)

# Evaluate your custom perceptron model
cust_y_pred_train, cust_sum_error = final(X_train, w_custom, y_train)
cust_y_pred_test, cust_sum_error = final(X_test, w_custom, y_test)

# Evaluate the sklearn Perceptron model
y_pred_train_sklearn = perceptron_sklearn.predict(X_train)
y_pred_test_sklearn = perceptron_sklearn.predict(X_test)

# Calculate accuracy and number of misclassifications for your custom perceptron model
cust_acc_train = accuracy_score(y_train, cust_y_pred_train)
accuracy_test_custom = accuracy_score(y_test, cust_y_pred_test)
cust_miss_train = (y_train != cust_y_pred_train).sum()
cust_miss_test = (y_test != cust_y_pred_test).sum()

# Calculate accuracy and number of misclassifications for the sklearn Perceptron model
accuracy_train_sklearn = accuracy_score(y_train, y_pred_train_sklearn)
accuracy_test_sklearn = accuracy_score(y_test, y_pred_test_sklearn)
miss_train_sklearn = (y_train != y_pred_train_sklearn).sum()
miss_test_sklearn = (y_test != y_pred_test_sklearn).sum()

# Print and compare the results
print("Custom Perceptron Model:")
print("Dataset 1:")
print("training Accuracy:", cust_acc_train)
print("test Accuracy:", accuracy_test_custom)
print("misclassifications (training):", cust_miss_train)
print("misclassifications (test):", cust_miss_test)

print("\nsklearn Perceptron Model:")
print("Dataset 1:")
print("training Accuracy:", accuracy_train_sklearn)
print("test Accuracy:", accuracy_test_sklearn)
print("misclassifications (training):", miss_train_sklearn)
print("misclassifications (test):", miss_test_sklearn)


#####

df = pd.read_csv('Dataset_2.csv')
cols = df.columns
x2 = df.iloc[:,0:ftrs].values
y2 = df['y'].values


std_scl = StandardScaler()
x_std2 = std_scl.fit_transform(x2[:,0:ftrs])
ones = np.ones(len(x_std2[:,0]),dtype=float)
Astd_2 = np.column_stack((ones,x_std2))

print('\nsimilarly for dataset 2: ')
y_pred_dataset2,sum_error=final(Astd_2, w, y2)
X_train, X_test, y_train, y_test = train_test_split(Astd_2,y2,test_size=0.33,random_state=0)



###
### Question 5
###

w_custom, cust_error, t_custom = sgd(X_train, np.ones(len(X_train[0])) * 0.1, y_train, 0.0001)

# Create and train the sklearn Perceptron model
perceptron_sklearn = Perceptron(max_iter=100, eta0=0.0001, random_state=0)
perceptron_sklearn.fit(X_train, y_train)

# Evaluate your custom perceptron model
cust_y_pred_train, cust_sum_error = final(X_train, w_custom, y_train)
cust_y_pred_test, cust_sum_error = final(X_test, w_custom, y_test)

# Evaluate the sklearn Perceptron model
y_pred_train_sklearn = perceptron_sklearn.predict(X_train)
y_pred_test_sklearn = perceptron_sklearn.predict(X_test)

# Calculate accuracy and number of misclassifications for your custom perceptron model
cust_acc_train = accuracy_score(y_train, cust_y_pred_train)
accuracy_test_custom = accuracy_score(y_test, cust_y_pred_test)
cust_miss_train = (y_train != cust_y_pred_train).sum()
cust_miss_test = (y_test != cust_y_pred_test).sum()

# Calculate accuracy and number of misclassifications for the sklearn Perceptron model
accuracy_train_sklearn = accuracy_score(y_train, y_pred_train_sklearn)
accuracy_test_sklearn = accuracy_score(y_test, y_pred_test_sklearn)
miss_train_sklearn = (y_train != y_pred_train_sklearn).sum()
miss_test_sklearn = (y_test != y_pred_test_sklearn).sum()

# Print and compare the results
print("Custom Perceptron Model:")
print("Dataset 2:")
print("training Accuracy:", cust_acc_train)
print("test Accuracy:", accuracy_test_custom)
print("misclassifications (training):", cust_miss_train)
print("misclassifications (test):", cust_miss_test)

print("\nsklearn Perceptron Model:")
print("Dataset 2:")
print("training Accuracy:", accuracy_train_sklearn)
print("test Accuracy:", accuracy_test_sklearn)
print("misclassifications (training):", miss_train_sklearn)
print("misclassifications (test):", miss_test_sklearn)
