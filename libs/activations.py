''' 
This file defines a helper class called Activation, which can be used to define both the non-linearity and its derivative for a network.
Additionally, all activation functions used in this notebook are defined here
'''

import numpy as np

class Activation:
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
        
    def activate(self, z):
        return self.activation(z)
    
    def derive(self, z):
        if z is not None:
            return self.derivative(z)
        else:
            print('No Derivative Defined!')
            return None


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


