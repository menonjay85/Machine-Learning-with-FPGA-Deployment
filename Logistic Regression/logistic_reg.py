# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:39:06 2024

@author: menon
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Define sigmoid function
def sigmoid(p):
    return 1.0 / (1.0 + np.exp(-np.clip(p, -250, 250)))

# Cross-entropy cost function
def cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Define activation function
def activation(Z):
    return sigmoid(Z)

# Logistic regression function
def log_reg(X, y, alpha, epochs):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        for i in range(X.shape[0]):
            z = np.dot(X[i], weights)
            prediction = activation(z)
            error = prediction - y[i]
            weights -= alpha * error * X[i]
    return weights

# One-hot encoding
def onehoty(y):
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y.reshape(len(y), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

# Logistic regression for multiclass classification
def log_reg_multiclass(X, y, alpha, epochs, num_classes):
    y_encoded = onehoty(y)
    weights_all = np.zeros((num_classes, X.shape[1]))

    for c in range(num_classes):
        y_c = y_encoded[:, c]
        weights_c = log_reg(X, y_c, alpha, epochs)
        weights_all[c, :] = weights_c

    return weights_all

# Argmax function for prediction
def argmax(Z):
    return np.argmax(Z, axis=1)

def predict_multiclass(X, weights):
    # Compute the probability for each class
    probabilities = sigmoid(np.dot(X, weights.T))
    # Use argmax to find the class with the highest probability
    predictions = argmax(probabilities)
    return predictions

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, 0:4]
y = iris.target

# Standardization
stan_scale = StandardScaler()
X_ss = stan_scale.fit_transform(X)

# Stacking column stack (intercept)
Nobs = X_ss.shape[0]
I = np.ones(Nobs, dtype=float)
X_ss = np.column_stack((I, X_ss))

# Binary classification
y_bin = y[y != 2]
weights = log_reg(X_ss[y != 2], y_bin, alpha=0.01, epochs=1000)
print("Binary Classification Weights:", weights)

# Multiclass classification
num_classes = len(np.unique(y))
weights_multiclass = log_reg_multiclass(X_ss, y, alpha=0.01, epochs=1000, num_classes=num_classes)
print("Multiclass Classification Weights:", weights_multiclass)

# Example: Making predictions on the standardized Iris dataset
predictions = predict_multiclass(X_ss, weights_multiclass)
print("Predictions:", predictions)


### Canned algorithm
num_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X_ss, y, test_size=0.3, random_state=42)
custom_predictions = predict_multiclass(X_test, weights_multiclass)

# scikit learn model
sklearn_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
sklearn_model.fit(X_train[:, 1:], y_train)
sklearn_predictions = sklearn_model.predict(X_test[:, 1:])

# Comparison
print("\nCustom Model Performance:")
print(classification_report(y_test, custom_predictions))
print("\nScikit-Learn Model Performance:")
print(classification_report(y_test, sklearn_predictions))
