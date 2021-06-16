#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:11:53 2021

@author: chris
"""
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from neural_network import BinaryClassificationShallowNetwork

# load sample dataset from sklearn
data = load_breast_cancer()
X = data.data # matrix containing the features
Y = data.target # vectory containing the true labels (binary classification)

# split data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# change shape of data to fit neural network
X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))

# parameter variables
num_features = X_train.shape[0]
num_training_samples = X_train.shape[1]
alpha = 0.001
binary_net = BinaryClassificationShallowNetwork(X=X_train, Y=y_train, 
                                                M=num_training_samples, input_size=num_features, 
                                                num_hidden_layers=15, 
                                                epochs=15000, 
                                                learning_rate=alpha)
binary_net.train() # train model

# get our predictions
y_pred = binary_net.predict(X_test, y_test)
y_test = np.squeeze(y_test) # convert matrix to vector for comparison

# get number of correct predictions
acc = np.sum([1 if y_test[i] == y_pred[i] else 0 for i in range(len(y_test))]) / len(y_test)
print(f"Accuracy is: {acc}")

    