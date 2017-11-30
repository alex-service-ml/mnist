from activations import *
from cost import *
import numpy as np
from misc import shuffle

''' 
    DropoutNetwork builds on what the Cross Entropy Network accomplishes by adding the ability to enable dropout. In the interest of simplicity, it's a boolean flag that causes 20% of inputs to be dropped and 50% of all other hidden layer activations to be dropped, as was used in the original Dropout paper by Hinton et. al: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    To support different dropout ratios would be trivial; simply update the constructor to accept a list of probabilities which has the same length as the number of layers (including input).
    Additionally, this network now captures and returns both training and testing accuracy while training, which can then be readily graphed.
'''
class DropoutNetwork:
    
    def __init__(self, *layers, activations=None, cost=Cost(cross_entropy_cost, cross_entropy_cost_prime), dropout=True):
        # assert activation is not None
        self.layers = [l for l in layers]
        
        if activations is not None:
            assert len(activations) == len(layers) - 1
            self.activations = [a for a in activations]
        else:
            self.activations = [Activation(sigmoid, sigmoid_prime) for i in range(len(layers) - 1)]
            
        self.W = [np.random.rand(layers[i], layers[i+1]) * 0.001 for i in range(len(layers) - 1)]  
        self.B = [np.random.rand(1, layers[i+1]) for i in range(len(layers) - 1)]  # Input layer doesn't have bias
        
        if dropout:
            self.drop = [0.8] + [0.5 for l in range(len(layers) - 2)] + [1] # 
        else:
            self.drop = [1.0 for l in range(len(layers))]
        self.cost = cost
    
    def feedforward(self, x, test=False):
        masks = []
        output = []
        a = x
        
        if test == False:
            masks += [dropout_mask(x.shape, threshold=1-self.drop[0])]
            a = np.multiply(a, masks[0])
        # Experiment with scaling inputs just as we do with activations
        
        output += [a]
        z_products = []
        
        for b, w, p, activation in zip(self.B, self.W, self.drop[1:], self.activations):
            if test == False:  # Only scale activations if we're testing
                p = 1
                
            z = np.dot(a, w) + b
            a = p * activation.activate(z)
            if test == False:
                masks += [dropout_mask(z.shape, threshold=1-p)]
                output.append(np.multiply(a, masks[-1]))
            else:
                output.append(a)
            z_products.append(z)
        return output, z_products, masks

    def train(self, images, labels, batch_size=200, learning_rate=0.01, epochs=10, test_epoch=5, test_images=None, test_labels=None, l2_lambda=0, l1_lambda=0):
        assert len(images) == len(labels)
        training_cost = []
        output = []
        for i in range(epochs):
            shuffle(images, labels)
            # Divide training data into batches
            img_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
            label_batches = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
            n_batches = len(img_batches)


            for imgs, lbls in zip(img_batches, label_batches):
                a, z, masks = self.feedforward(imgs)

                # Calculate Output Error
                # dL = np.multiply(self.cost.calculate_derivative(lbls, a[-1]),self.activations[-1].derive(z[-1]))  # MSE Cost Error
                l2_loss = (1 - learning_rate * l2_lambda / batch_size)
                l1_loss = - learning_rate * l1_lambda / batch_size
                dL = self.cost.calculate_derivative(lbls, a[-1])  # Cross Entropy doesn't have hadamard product  # CE Cost Error
                # Update biases and weights for output layer using output error
                dBL = np.reshape(np.sum(dL, axis=0), (1,dL.shape[1]))
                dWL = np.dot(a[-2].T, dL)
                # Being verbose to show that we're building a list of deltas for each layer in the network (excluding input layer)
                dBl = [dBL]
                dWl = [dWL]
                dl = dL
                
                # Calculate Hidden Error
                for l in range(len(a) - 2, 0, -1):  # -1 for zero-indexing, -1 more because output layer already calculated
                    dl = np.multiply(np.dot(dl, self.W[l].T), np.multiply(self.activations[l-1].derive(z[l-1]), masks[l]))
                    dBl.insert(0, np.reshape(np.sum(dl, axis=0), (1, dl.shape[1])))
                    dWl.insert(0, np.dot(a[l-1].T, dl))

                # Update weights and biases
                for j in range(len(self.W)):
                    self.W[j] = l2_loss * self.W[j] + l1_loss * np.sign(self.W[j]) - (learning_rate / batch_size) * dWl[j]
                    self.B[j] -= (learning_rate / batch_size) * dBl[j]


            # Cost after training this epoch
            a, z, masks = self.feedforward(images, test=True)
            l2_cost = np.sum([np.sum(w ** 2) for w in self.W]) * l2_lambda / (2 * len(a))
            l1_cost = l1_lambda * np.sum([np.sum(np.abs(w)) for w in self.W]) / len(a)
            training_cost.append(self.cost.calculate(labels, a[-1]) + l2_cost + l1_cost)
            result = [training_cost[-1]]

            result += [self.test(images, labels) / len(images)]
            if test_images is not None and test_labels is not None and (i + 1) % test_epoch == 0:
                result += [self.test(test_images, test_labels) / len(test_images)]
            else:
                result += [None]
            print(result)
            output.append(result)
        return np.asarray(output)
    
    def test(self, test_images, test_labels):
        img = test_images
        a, z, masks = self.feedforward(img, test=True)
        a[-1].shape
        prediction = np.argmax(a[-1], axis=1)
        truth = np.argmax(test_labels, axis=1)
        result = np.sum([p == i for p, i in zip(prediction, truth)])
        return result
