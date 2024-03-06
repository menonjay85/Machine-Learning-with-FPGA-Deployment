# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:37:33 2024

@author: menon
"""

##
# Question 4
##
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('regressionprob1_train0.csv')
X = df.iloc[:,0:4].values
y = df['F'].values
Size = len(y)
I = np.ones(Size,float)
A = np.column_stack((I,X))
A_TA = np.matmul(np.transpose(A),A)
A_Ty = np.matmul(np.transpose(A),y)

w = np.matmul(np.linalg.inv(A_TA),A_Ty)
ypred = np.matmul(A,w)
r2_1= r2_score(y,ypred)
w_new = np.linalg.solve(A_TA,A_Ty)
ypred_new = np.matmul(A,w_new)
r2_2 = r2_score(y,ypred_new)
print('Least Squares: Training weights =>', w, '\n R2 =>', r2_1)
print('Intercept =>', w[0], '\n')
print('Linear Solver: Solve weights =>', w_new, '\n R2 =>', r2_2)
print('Intercept =>', w_new[0], '\n')


test = pd.read_csv('regressionprob1_test0.csv')
test_x = test.iloc[:,0:4].values
test_A = np.column_stack((np.ones(len(test_x)), test_x))
y_test1 = np.matmul(test_A, w)
y_test2 = np.matmul(test_A, w_new)
r2_3 = r2_score(test['F'].values,y_test1)
r2_4 = r2_score(test['F'].values,y_test2)
print('Test R2 (compact notation) =>', r2_3)
print('Test R2 (linear solver) =>', r2_4, '\n')


##
# Question 5
##
w_init = np.zeros(A.shape[1])
alpha = 0.001
num_iterations = 1000

# Gradient Descent
for _ in range(num_iterations):
    ypred_gd = np.dot(A, w_init)
    errs = ypred_gd - y
    cf = 2 * np.dot(A.T, errs) / len(y)
    w_init -= alpha * cf

r2_5 = r2_score(y, np.dot(A, w_init))
print('Gradient Descent weights =>', w_init[1:], '\nIntercept =>', w_init[0])
print('R2 Score (Gradient Descent) =>', r2_5, '\n')


##
# Question 6
##

df = pd.read_csv('regressionprob1_train0.csv')
X = df.iloc[:, 0:4].values
y = df['F'].values

model = LinearRegression()
model.fit(X, y)
w_canned = np.append(model.intercept_, model.coef_)
ypred_canned = model.predict(X)
r2_6 = r2_score(y, ypred_canned)

print('Canned Linear Regression weights =>', w_canned[1:], '\nIntercept =>', w_canned[0])
print('R2 Score (Canned Linear Regression) =>', r2_6)
