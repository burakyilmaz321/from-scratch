'''
Basic neural network. Single layer perceptron.
'''
import numpy as np

def sigm(x):
    return 1 / (1 + np.exp(-x))

def feedforward(x, w1, w2, y):
    a1 = sigm(x * w)
    a2 = sigm(a1 * w2)
    e = (y - a2)**2 / 2
    return (a2, e)
    
def backprop(a1, a2, x, w1, w2, y):
    delta2 = (a - y) * a * (1 - a)
    delta1 = w2 * delta2 * a * (1 - a)
    w2_updated = w2 - (delta2 * a1)
    w2_updated = w - (delta1 * x)
    return w1_updated, w2_updated

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

