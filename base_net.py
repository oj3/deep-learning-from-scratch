import sys, os
sys.path.append(os.pardir)
import numpy as np
from trainer import Trainer
from ch4 import numerical_gradient
from dataset.mnist import load_mnist

class NeuralNetwork:    
    def check(self, size = 3):
        chk = GradientCheck(self, size)
        chk.check()

class GradientCheck:
    def __init__(self, network, size = 3, normalize = True, one_hot_label = True):
        self.network = network
        self.size = size
        self.normalize = normalize
        self.one_hot_label = one_hot_label

        self.grad_numerical = None
        self.grad_backprop = None
        self.diff = None

    def check(self):
        # Load MNIST data
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize = self.normalize, one_hot_label = self.one_hot_label)
        x_batch = x_train[:self.size]
        t_batch = t_train[:self.size]

        self.grad_numerical = self.network.numerical_gradient(x_batch, t_batch)
        self.grad_backprop = self.network.gradient(x_batch, t_batch)

        # Compare
        self.diff = {}
        for key in self.grad_numerical.keys():
            self.diff[key] = np.average( np.abs(self.grad_backprop[key] - self.grad_numerical[key]) )

        self.show_diff()

    def show_diff(self):
        for key in self.diff.keys():
            diff = self.diff[key]

            text = key + ": "
            text += str(diff)
            print(text)

            grad_n = np.average(np.abs(self.grad_numerical[key]))
            text = str(diff / grad_n * 100) + "% of "
            text += str(grad_n) + " [grad_numerical]"
            print("    --> " + text)

            grad_b = np.average(np.abs(self.grad_backprop[key]))
            text = str(diff / grad_b * 100) + "% of "
            text += str(grad_b) + " [grad_backprop]"
            print("    --> " + text)