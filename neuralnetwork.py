# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:07:12 2020

@author: César
"""

import numpy as np
from numpy import exp

def relu(x):
    return x if x > 0 else 0

def sigmoid(x):
    return 1 / (1 + exp(-x))

def hyp_tangent(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def inv_relu(x):
    return 1 if x > 0 else 0

def inv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def inv_hyp_tangent(x):
    return 1 / hyp_tangent(x)

def activation(x, function):
    if function == "relu":
        for i in range(len(x[0])):
            x[0][i] = relu(x[0][i])
        return x
    elif function == "sigmoid":
        for i in range(len(x[0])):
            x[0][i] = sigmoid(x[0][i])
        return x
    elif function == "hyp_tangent":
        for i in range(len(x[0])):
            x[0][i] = hyp_tangent(x[0][i])
        return x
    else:
        return x

def deactivation(x, function):
    if function == "relu":
        for i in range(len(x)):
            x[i] = inv_relu(x[i])
        return x
    elif function == "sigmoid":
        for i in range(len(x)):
            x[i] = inv_sigmoid(x[i])
        return x
    elif function == "hyp_tangent":
        for i in range(len(x)):
            x[i] = inv_hyp_tangent(x[i])
        return x
    else:
        return x  

class NeuralNetwork:
    
    def __init__(self, input_shape, output_shape, hidden_shape = None, hidden_functions = "relu",
                 output_function = "relu"):
        
        self.hidden_shape = [input_shape] if hidden_shape == None else hidden_shape
        
        if type(hidden_functions) == str:
            hidden_functions = [hidden_functions]
        for _ in range(len(self.hidden_shape) - len(hidden_functions)):
            hidden_functions.append("relu")
                
        self.functions = hidden_functions + [output_function]
        
        self.weights = []
        self.bias = []
        self.weights.append(np.random.rand(input_shape, self.hidden_shape[0]))
        self.bias.append(np.random.rand(1, self.hidden_shape[0]))
        for i in range(1, len(self.hidden_shape)):
            self.weights.append(np.random.rand(self.hidden_shape[i - 1], self.hidden_shape[i]))
            self.bias.append(np.random.rand(1, self.hidden_shape[i]))
        self.weights.append(np.random.rand(self.hidden_shape[-1], output_shape))
        self.bias.append(np.random.rand(1, output_shape))
        
        self.error_type = "mse"
        
        self.enable_grad = True
        self.register = []
    
    def forward(self, x):
        x = np.array(x)
        if self.enable_grad:
            self.register = []
            self.register.append(x)
        z = x
        for i in range(len(self.weights)):
            z = z @ self.weights[i] + self.bias[i]
            if self.enable_grad:
                self.register.append(z[0].copy())
            z = activation(z, self.functions[i])
            if self.enable_grad:
                self.register.append(z[0])
        return z

    def loss_error(self, y_pred, y_real, loss_function = "mse"):
        self.error_type = loss_function
        if loss_function == "mse":
            deriv_e = np.zeros(len(y_pred))
            error = 0
            for i in range(len(y_pred)):
                deriv_e[i] = y_pred[i]-y_real[i]
                error += 0.5 * (y_pred[i]-y_real[i]) ** 2
            if self.enable_grad and len(self.register) < 2 * len(self.weights) + 2:
                self.register.append(deriv_e)
            return error
    
    def backpropagation(self, learning_rate = 1e-5):
        lr = learning_rate
        if self.error_type == "mse":
            deriv = self.register[-1]
        deriv_f = deactivation(self.register[-3].copy(), self.functions[-1])
        deriv = np.asmatrix(deriv * deriv_f)
        for i in range(len(self.weights),0,-1):
            output = np.asmatrix(self.register[2*i-2])
            store_weights = self.weights[i-1].copy()
            self.bias[i-1] = np.asarray(self.bias[i-1] - lr * deriv)
            self.weights[i-1] = np.asarray(self.weights[i-1] - lr * output.T @ deriv)
            deriv = deriv @ store_weights.T
            deriv_f = deactivation(self.register[2 * i].copy(), self.functions[i-2])
            deriv = np.asmatrix(deriv)

        
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_len = 3
    y_len = [4, 3]
    z_len = 2
    nn = NeuralNetwork(x_len, z_len, y_len, ["relu", "relu"], "sigmoid")
    x = [0.1, 0.5, 0.4]
    z_real = [0.5, 0.3]
    error = []
    z1 = []
    z2 = []
    for _ in range(50):
        z = nn.forward(x)[0]
        z1.append(z[0])
        z2.append(z[1])
        error.append(nn.loss_error(z, z_real))
        nn.backpropagation(0.01)
        print(x, z, z_real)
    plt.plot(error)
    plt.plot(z1)
    plt.plot(z2)
    plt.figure()