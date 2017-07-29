'''
Basic neural network. Single layer perceptron.
'''
import numpy as np

def sigm(x):
    return 1 / (1 + np.exp(-x))

def feedforward(x, w, y):
    a = sigm(x * w)
    e = (y - a)**2 / 2
    return (a, e)
    
def backprop(a, x, w, y):
    delta = (a - y) * a * (1 - a)
    w_updated = w - (delta * x)
    return w_updated

def train(x, w, y, epoch):
    for _ in range(epoch):
        a, e = feedforward(x, w, y)
        w = backprop(a, x, w, y)
        if _ % (epoch / 10) == 0:
            print('-> epoch #' + str(_))
            print('    Error : {:.4f}'.format(e))
            print('    Output: {:.4f}'.format(a))
            print('    Weight: {:.4f}'.format(w))
            print('')
    
    print('Final error : {:.4f}'.format(e))
    print('Final output: {:.4f}'.format(a))
    print('Final weight: {:.4f}'.format(w))

