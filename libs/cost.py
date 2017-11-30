''' 
This file defines a helper class called Cost, which can be used to define both the cost function and its derivative for a network.
Additionally, all cost functions used in this notebook are defined here
'''

import numpy as np

class Cost:
    def __init__(self, calculation, derivative):
        self.calculation = calculation
        self.derivative = derivative

    def calculate(self, y, y_pred):
        return self.calculation(y, y_pred)

    def calculate_derivative(self, y, y_pred):
        return self.derivative(y, y_pred)

def mse_cost(y, y_pred):
    return 0.5 * (1 / y.shape[0]) * np.sum(np.abs(y - y_pred) ** 2)

def cross_entropy_cost(y, y_pred):
    ce = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    return - np.sum(ce) / len(ce)

# Most cost functions are designed so that their derivative is this same equation
def mse_cost_prime(y, y_pred):
    return y_pred - y

def cross_entropy_cost_prime(y, y_pred):
    return y_pred - y