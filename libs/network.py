from activations import *
from cost import *
import numpy as np
from misc import shuffle

class Network:
    
    def __init__(self, *layers, activation=Activation(sigmoid, sigmoid_prime), cost=Cost(mse_cost, mse_cost_prime)):
        # assert activation is not None
        self.layers = [l for l in layers]
        self.W = [np.random.rand(layers[i], layers[i+1]) * 0.001 for i in range(len(layers) - 1)]
        self.B = [np.random.rand(1, layers[i+1]) for i in range(len(layers) - 1)]  # Input layer doesn't have bias
        self.activation = activation
        self.cost = cost
        
    def feedforward(self, x):
        a = x  # re-label input to simplify loop
        activations = [a]
        z_products = []
        for b, w in zip(self.B, self.W):
            z = np.dot(a, w) + b
            a = self.activation.activate(z)
            activations.append(a)
            z_products.append(z)
        return activations, z_products
    
    def train(self, images, labels, batch_size=200, learning_rate=0.01, epochs=10, test_epoch=5, test_images=None, test_labels=None):
        assert len(images) == len(labels)
        training_cost = []
        for i in range(epochs):
            shuffle(images, labels)
            # Divide training data into batches
            img_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
            label_batches = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
            n_batches = len(img_batches)

            for imgs, lbls in zip(img_batches, label_batches):
                a, z = self.feedforward(imgs)

                # Calculate Output Error
                dL = np.multiply(self.cost.calculate_derivative(lbls, a[-1]),self.activation.derive(z[-1]))

                # Update biases and weights for output layer using output error
                dBL = np.reshape(np.sum(dL, axis=0), (1,dL.shape[1]))
                dWL = np.dot(a[-2].T, dL)
                # Being verbose to show that we're building a list of deltas for each layer in the network (excluding input layer)
                dBl = [dBL]
                dWl = [dWL]
                dl = dL
                
                # Calculate Hidden Error
                for l in range(len(a) - 2, 0, -1):  # -1 for zero-indexing, -1 more because output layer already calculated
                    dl = np.multiply(np.dot(dl, self.W[l].T), self.activation.derive(z[l-1]))
                    dBl.insert(0, np.reshape(np.sum(dl, axis=0), (1, dl.shape[1])))
                    dWl.insert(0, np.dot(a[l-1].T, dl))

                # Update weights and biases
                for j in range(len(self.W)):
                    self.W[j] -= (learning_rate / batch_size) * dWl[j]
                    self.B[j] -= (learning_rate / batch_size) * dBl[j]


            # Cost after training this epoch
            a, z = self.feedforward(images)
            training_cost.append(self.cost.calculate(labels, a[-1]))
            result = str(training_cost[-1])

            if test_images is not None and test_labels is not None and (i + 1) % test_epoch == 0:
                result += ' ' + str(self.test(test_images, test_labels)) +  ' / 10000'
            print(result)
        return training_cost
    
    def test(self, test_images, test_labels):
        img = test_images
        a, z = self.feedforward(img)
        a[-1].shape
        prediction = np.argmax(a[-1], axis=1)
        truth = np.argmax(test_labels, axis=1)
        result = np.sum([p == i for p, i in zip(prediction, truth)])
        return result
