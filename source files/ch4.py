import sys, os
sys.path.append(os.pardir)
import datetime as dt
import numpy as np
from dataset.mnist import load_mnist
from ch3 import sigmoid, softmax


def test_minibatch():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, one_hot_label = True)

    print("x_train.shape = " + str(x_train.shape))
    print("t_train.shape = " + str(t_train.shape))
    
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    print("x_batch.shape = " + str(x_batch.shape))
    print("t_batch.shape = " + str(t_batch.shape))

def mean_square_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error_by_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) /batch_size

def numerical_diff(f, x):
    dx = 1e-4
    return (f(x+dx) - f(x-dx)) / (2*dx)

def numerical_gradient(f, x):
    if x.ndim == 1:
        return grad_on_vector(f, x)
    elif x.ndim == 2:
        return grad_on_matrix(f, x)

def grad_on_vector(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        x_value = x[i]
        
        x[i] = x_value + h
        fxh1 = f(x)        
        x[i] = x_value - h
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = x_value
    
    return grad

def grad_on_matrix(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        index0 = int(i / x.shape[1])
        index1 = i % x.shape[1]
        x_value = x[index0, index1]
        
        x[index0, index1] = x_value + h
        fxh1 = f(x)        
        x[index0, index1] = x_value - h
        fxh2 = f(x)
        
        grad[index0, index1] = (fxh1 - fxh2) / (2 * h)
        x[index0, index1] = x_value
    
    return grad

def gradient_descent(f, init_x, lr = 0.1, step_num = 100):
    x = init_x
        
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x
    
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                    weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # x: input, t: answers
    def loss(self, x, t):
        y = self. predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = predict(x)
        prediction = np.argmax(y, axis = 1)
        answer = np.argmax(t, axis = 1)
        
        return np.sum(prediction == answer) / float(x.shape[0])
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

class NN:
    def __init__(self, iteration = 10000, batch_size = 100):
        self.iter_num = iteration
        self.batch_size = batch_size
        self.learning_rate = 0.1
        
        self.init_network()
    
    # Reset learning history
    def init_network(self):
        self.network = TwoLayerNet(28 * 28, 50, 10)
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
    def load(self):
        (self.x_train, self.t_train),\
        (self.x_test, self.t_test) = \
            load_mnist(normalize = True, one_hot_label = True)
    
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(int(self.train_size / self.batch_size), 1)
    
    def learn(self):        
        for i in range(self.iter_num):
            # Choose batch set
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            # Gradient calculation
            grad = self.network.numerical_gradient(x_batch, t_batch)
            
            # Update parameters
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.network.params[key] -= self.learning_rate * grad[key]
            
            # Record history
            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)
            
            if (i + 1) % 5 == 0:
                current_time = dt.datetime.today()
                print(str(current_time) + " ---- loss[" + str(i) + "] = " + str(loss))
            
            if i % self.iter_per_epoch == 0:
                train_acc = self.network.loss(self.x_train, self.t_train)
                test_acc = self.network.loss(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                
                print("(train_acc, test_acc) = (" + str(train_acc) + ", "+ str(test_acc) + ")")