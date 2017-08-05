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

    def __init__(self, hidden_unit, alpha):
        self.hidden_unit = hidden_unit
        self.alpha = alpha

    def report(self, x, y, e, epoch, iter):
        if iter % (epoch / 10) == 0:
            print('-> epoch #' + str(iter))
            print('    Error : {:.4f}'.format(np.mean(e)))
            print('    Output: ', [i for i in zip(x.flatten(), y.flatten())])
            print('')

    def feedforward(self, a1, w1, w2, y):
        a2 = ActivationFunctions.relu(np.dot(a1, w1))
        a3 = ActivationFunctions.sigm(np.dot(a2, w2))
        e = (y - a3)**2 / 2
        return (a2, a3, e)
        
    def backprop(self, a1, a2, a3, w1, w2, y):
        delta2 = (a3 - y) * ActivationFunctions.sigm_d(a3)
        delta1 = np.dot(delta2, w2.T) * ActivationFunctions.relu_d(a2)
        w2 -= self.alpha * np.dot(a2.T, delta2)
        w1 -= self.alpha * np.dot(a1.T, delta1)
        return w1, w2

    def fit(self, a1, y, epoch, silent=False):
        input_unit = a1.shape[1]
        w1 = np.random.random((input_unit, self.hidden_unit))
        w2 = np.random.random((self.hidden_unit, 1))
        for _ in range(epoch):
            a2, a3, e = self.feedforward(a1, w1, w2, y)
            w1, w2 = self.backprop(a1, a2, a3, w1, w2, y)
            if not silent: 
                self.report(a3, y, e, epoch, _)

        print('Final error : {:.4f}'.format(np.mean(e)))
        print('Final output: ', [i for i in zip(a3.flatten(), y.flatten())])

def main():
    '''
    Main function should initialize input space, output space and the network
    Input and output should be numpy arrays.
    Network instance must be initialized with topology, learning rate, activation function 
    To train the network, input and output arrays must be passed as an argument 
    as well as the desired number of training epochs
    '''
    X = np.array([0,0,1,1,0,1,0,1,1,1,1,1]).reshape((4, 3))
    y = np.array([0, 1, 1, 0]).reshape((4, 1))
    net = NeuralNetwork(5, 0.5)
    net.fit(X, y, 1000)

if __name__ == "__main__":
    main()
