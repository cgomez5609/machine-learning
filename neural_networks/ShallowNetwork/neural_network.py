#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:30:10 2021

@author: chris
"""
import numpy as np

class BinaryClassificationShallowNetwork:
    def __init__(self, X, Y, M, input_size, num_hidden_layers, epochs, learning_rate):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.M = M
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.check_XY_shapes()
        self.parameters = self.__initialize_weights_bias()
        self.cache = dict()
        self.A2 = None
        self.grads = dict()
        self.costs = list()
        
    def check_XY_shapes(self):
        assert self.X.shape == (self.input_size, self.M), "X-shape should be nxm where n is the input size and m is the number of training examples"
        assert self.Y.shape == (1, self.M), "Y-shape should be 1xm where m is the number of training examples"
    
    def get_layer_sizes(self):
        input_layer = self.X.shape[0]
        hidden_layer = self.num_hidden_layers
        output_layer = self.Y.shape[0]
        return (input_layer, hidden_layer, output_layer)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def compute_cost(self):
        logs = np.dot(np.log(self.A2), self.Y.T) + np.dot((1-self.Y), np.log((1-self.A2).T))
        cost = (logs / self.M) *-1
        cost = float(np.squeeze(cost))
    
        return cost
    
    def __initialize_weights_bias(self):
        (input_l_size, hidden_l_size, output_l_size) = self.get_layer_sizes()
        W1 = np.random.randn(hidden_l_size, input_l_size) * 0.01
        b1 = np.zeros((hidden_l_size, 1))
        W2 = np.random.randn(output_l_size, hidden_l_size) * 0.01
        b2 = np.zeros((output_l_size, 1))
        parameters = {"W1":W1,
                      "b1":b1,
                      "W2":W2,
                      "b2":b2}
        return parameters
        
    def __forward(self):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        
        Z1 = np.dot(W1, self.X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        
        assert(A2.shape == (1, self.X.shape[1]))
        self.cache = {"Z1": Z1,
               "A1": A1,
               "Z2": Z2,
               "A2": A2}
        self.A2 = A2
        
    def __backpropagation(self):
        W2 = self.parameters["W2"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        g1 = (1 - np.power(A1, 2))
        
        dZ2 = A2 - self.Y
        dW2 = np.dot(dZ2, A1.T) / self.M
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.M
        dZ1 = np.dot(W2.T, dZ2) * g1
        dW1 = np.dot(dZ1, self.X.T) / self.M
        db1 = np.sum(dZ1, axis=1, keepdims=True) / self.M
        
        self.grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        self.__update_parameters()
        
    def __update_parameters(self):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        
        dW1 = self.grads["dW1"]
        db1 = self.grads["db1"]
        dW2 = self.grads["dW2"]
        db2 = self.grads["db2"]
    
        W1 = W1 - (self.learning_rate * dW1)
        b1 = b1 - (self.learning_rate * db1)
        W2 = W2 - (self.learning_rate * dW2)
        b2 = b2 - (self.learning_rate * db2)
        
        self.parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
    
    def train(self, print_at_epochs=10):
        for i in range(self.epochs):
            self.__forward()
            self.costs.append(self.compute_cost())
            self.__backpropagation()
            
            if i % print_at_epochs == 0:
                print(f"cost at epoch {i} is: {self.costs[i]}")
                
    def __test_forward(self, X_test):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        
        Z1 = np.dot(W1, X_test) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        
        assert(A2.shape == (1, X_test.shape[1]))
        return A2
                
    def predict(self, X_test, y_test, threshold=0.5):
        predictions = self.__test_forward(X_test)
        predictions = np.where(predictions>threshold, 1, 0)
        return np.squeeze(predictions)
        
        