#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 06:07:29 2019

@author: shahnawaz
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

class mlp:
     # layers : number of nodes in each layers, 
     # lr : learning rate for individual layers, 
     # tr_func : transfer function for individual layers (0 - tanh, 1 - sigmoid, 2 - linear)
    def __init__(self, layers, lr, tr_func):
        self.nl = len(layers)
        self.layers = layers
        self.lr = lr
        self.tr_func = tr_func
        self.w = {}
        k = 1
        np.random.seed(0)
        while k<self.nl: # Random Initialization of the weights
            self.w[k-1] = 4* np.random.rand(layers[k-1]+1, layers[k]) -2.0
            k = k+1
        self.net = {}
        self.out = {}
        
    def tr(self, X, n):
        if self.tr_func[n-1] == 0:
            return np.tanh(X)
        elif self.tr_func[n-1] == 1:
            return 1/(1+ np.exp(-X))
        else:
            return X
    
    def diff_tr(self, X, n):
        if self.tr_func[n-1] == 0:
            return 1 - self.tr(X, n)*self.tr(X, n)
        elif self.tr_func[n-1] == 1:
            return self.tr(X, n)*(1 - self.tr(X, n))
        else:
            return 1
    
    def forward(self, X):
        k = 0
        while k < self.nl:
            if k == 0:
                self.net[k] = X
                self.out[k] = self.net[k]
            else:
                biased_input = np.concatenate(([[1]], self.out[k-1]), axis=1)
                self.net[k] = np.dot(biased_input, self.w[k-1])
                self.out[k] = self.tr(self.net[k], k-1)
            k += 1
        return self.out[k-1]
    
    def train(self, X, Y):
        self.forward(X)
        
        # Backpropagation
        k = self.nl - 1
        delta = {}
        while k > 0: # Iterate layer backward
            delta_tmp = np.zeros((1, self.layers[k]))
            w_tmp = np.zeros_like(self.w[k-1])
            for node in range(self.layers[k]): # Iterate over all the nodes in the layer
                if k == self.nl - 1:
                    delta_tmp[0, node] = (Y[0, node] - self.out[k][0, node]) * self.diff_tr(self.net[k][0, node], k)
                else:
                    sum_delta = np.dot(delta[k+1], self.w[k][node])
                    delta_tmp[0, node] = sum_delta * self.diff_tr(self.net[k][0, node], k)
                    
            for i in range(np.shape(w_tmp)[0]): # Updation of the weights
                for j in range(np.shape(w_tmp)[1]):
                    if i == 0:
                        _out = 1
                    else:
                        _out = self.out[k-1][0, i-1]
                    _del = delta_tmp[0, j]
                    w_tmp[i,j] = self.lr[k-1]* _del * _out
                    
            self.w[k-1] += w_tmp
            delta[k] = delta_tmp
            k = k-1

def data(): # XOR Data
    X = [[1,1],[0,0],[1,0],[0,1]]
    Y = [0,0,1,1]
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    X, Y = data()
    Y=np.expand_dims(Y,1)
    
    epochs = 40
    err = []
   
    perceptron = mlp(layers = [np.shape(X)[1], 2, np.shape(Y)[1]], lr=[0.1, 0.01], tr_func=[2, 2])  # Initializing the Perceptron
    for epoch in range(epochs): # Looping through training epochs number of times
        for pattern in range(np.shape(X)[0]):
            x = np.expand_dims(X[pattern,:],1).T
            y = np.expand_dims(Y[pattern,:],1).T
            perceptron.train(x, y)
            err.append(np.mean(np.sqrt(np.square(np.round(perceptron.forward(x)) - y))))# Error of the network step wise
    
    print(perceptron.w)
    plt.plot(err)
    plt.savefig('error_plot.pdf')
            
