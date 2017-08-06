'''
Basic 3-layer neural network implementation. 
a1: input matrix
a2: hidden layer
a3: output
'''

import numpy as np

class ActivationFunctions(object):
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    def sigm_d(x):
        return x * (1 - x)

    def relu(x):
        return np.maximum(0, x)
    
    def relu_d(x):
        def f(x):
            if x > 0:
                return 1
            else:
                return 0
        return np.vectorize(f)(x)

class NeuralNetwork(object):
    '''
    Description: 3-Layer perceptron class. 
    Topology: A tuple. (#of input units, #of hidden units)
    Activation: 'sigm' or 'relu'
    Alpha: learning rate between 0 and 1
    '''
    activations = {
        'sigm': (ActivationFunctions.sigm, ActivationFunctions.sigm_d),
        'relu': (ActivationFunctions.relu, ActivationFunctions.relu_d)
    }

    def __init__(self, topology, activation, alpha):
        self.topology = topology
        self.activation = activation    
        self.alpha = alpha

    def report(self, x, y, e, epoch, iter):
        if iter % (epoch / 10) == 0:
            print('-> epoch #' + str(iter))
            print('    Error : {:.4f}'.format(np.mean(e)))
            #print('    Output: ', [i for i in zip(x.flatten(), y.flatten())])
            print('')

    def feedforward(self, a1, w1, w2, y):
        a2 = self.activations[self.activation][0](np.dot(a1, w1))
        a3 = self.activations['sigm'][0](np.dot(a2, w2))
        e = (y - a3)**2 / 2
        return (a2, a3, e)
        
    def backprop(self, a1, a2, a3, w1, w2, y):
        delta2 = (a3 - y) * self.activations['sigm'][1](a3)
        delta1 = np.dot(delta2, w2.T) * self.activations[self.activation][1](a2)
        w2 -= self.alpha * np.dot(a2.T, delta2)
        w1 -= self.alpha * np.dot(a1.T, delta1)
        return w1, w2

    def fit(self, X, y, epoch, silent=False):
        '''
        X, y: numpy 1-D array. Will be reshaped according to the network topology
        '''        
        
        n = len(X) // self.topology[0]
        # Input space
        X = np.append(X, [1] * n)
        a1 = X.reshape((n, self.topology[0] + 1), order='F')
        # Output space
        y = y.reshape((n, 1), order='F')
        # Weight space
        w1 = np.random.random((self.topology[0] + 1, self.topology[1]))
        w2 = np.random.random((self.topology[1], 1))
        # Train
        for _ in range(epoch):
            a2, a3, e = self.feedforward(a1, w1, w2, y)
            w1, w2 = self.backprop(a1, a2, a3, w1, w2, y)
            if not silent: 
                self.report(a3, y, e, epoch, _)

        print('Final error : {:.4f}'.format(np.mean(e)))
        #print('Final output: ', [i for i in zip(a3.flatten(), y.flatten())])

def main():
    '''
    Main function should initialize input space, output space and the network
    Input and output should be numpy arrays.
    Network instance must be initialized with topology, learning rate, activation function 
    To train the network, input and output arrays must be passed as an argument 
    as well as the desired number of training epochs
    '''
    X = np.array([0,0,1,1,0,1,0,1])
    y = np.array([0, 1, 1, 0])
    network_topology = (2, 4)
    net = NeuralNetwork(network_topology, 'sigm', 0.5)
    net.fit(X, y, 1000)

if __name__ == "__main__":
    main()
