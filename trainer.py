import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

class Trainer:
    def __init__(self, network, opt):
        self.network = network
        self.opt = opt

    def learn(self):
        if self.network is None:
            print("Please specify the network.")
            return
        if self.opt is None:
            print("Please specify the type of optimizer.")
            return
        
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize = True, one_hot_label = True)
                
        iterations = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        lerning_rate = 0.1
        
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        iterations_per_epoch = max(train_size / batch_size, 1)
        
        for i in range(iterations):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            grad = self.network.gradient(x_batch, t_batch)
            self.opt.update(self.network.params, grad)
            
            loss = self.network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            
            if (i % iterations_per_epoch == 0):
                train_acc = self.network.accuracy(x_train, t_train)
                test_acc = self.network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(train_acc, test_acc)