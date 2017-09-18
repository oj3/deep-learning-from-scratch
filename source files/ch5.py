import sys, os
sys.path.append(os.pardir)
import numpy as np
from base_net import NeuralNetwork
from trainer import Trainer
from layers import *
from ch4 import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist

def learn(opt):
    net = TwoLayerNet(784, 50, 10)
    trainer = Trainer(net, opt)
    trainer.learn()
    return
    
class TwoLayerNet(NeuralNetwork):
    def __init__(self, \
        input_size, hidden_size, output_size, \
        weight_init_std = 0.01):
        
        # Initialize weight parameters
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # Prepare layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    # x: input, t: teacher data
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # Set gradient parameter values
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        ans = np.argmax(y, axis = 1)
        
        ref = np.argmax(t, axis = 1) if t.ndim != 1 else t
        
        accuracy = np.sum(ans == ref) / float(x.shape[0])
        return accuracy 