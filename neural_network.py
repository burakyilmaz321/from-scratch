'''
Basic 3-layer neural network implementation. 
a1: input matrix
a2: hidden layer
a3: output
'''
import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_unit, alpha):
        self.hidden_unit = hidden_unit
        self.alpha = alpha
    
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    def sigm_d(x):
        return x * (1 - x)
    def report(a3, y, e, epoch, iter):
        if iter % (epoch / 10) == 0:
                        print('-> epoch #' + str(iter))
                        print('    Error : {:.4f}'.format(np.mean(e)))
                        print('    Output: ', [i for i in zip(a3.flatten(), y.flatten())])
                        print('')

    def feedforward(a1, w1, w2, y):
        a2 = sigm(np.dot(a1, w1))
        a3 = sigm(np.dot(a2, w2))
        e = (y - a3)**2 / 2
        return (a2, a3, e)
        
    def backprop(a1, a2, a3, w1, w2, y, alpha):
        delta2 = (a3 - y) * sigm_d(a3)
        delta1 = np.dot(delta2, w2.T) * sigm_d(a2)
        w2 -= alpha * np.dot(a2.T, delta2)
        w1 -= alpha * np.dot(a1.T, delta1)
        return w1, w2

    def train(a1, y, alpha, hidden_unit, epoch, silent):
        input_unit = a1.shape[1]
        w1 = np.random.random((input_unit, hidden_unit))
        w2 = np.random.random((hidden_unit, 1))
        for _ in range(epoch):
            a2, a3, e = feedforward(a1, w1, w2, y)
            w1, w2 = backprop(a1, a2, a3, w1, w2, y, alpha)
            if not silent: 
                report(a3, y, e, epoch, _)

        print('Final error : {:.4f}'.format(np.mean(e)))
        print('Final output: ', [i for i in zip(a3.flatten(), y.flatten())])

def main():
    net = NeuralNetwork()
    a1 = np.array([0,0,1,1,0,1,0,1,1,1,1,1]).reshape((4, 3))
    y = np.array([0, 1, 1, 0]).reshape((4, 1))
    train(a1, y, 0.9, 5, 10000, True)

if __name__ == "__main__":
    main()
