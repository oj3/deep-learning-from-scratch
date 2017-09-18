import sys
import matplotlib.pylab as plt
import numpy as np
from ch4 import numerical_diff, numerical_gradient

def set_lim(min, max):
    plt.ylim(min, max)
    
def set_lim_abs(lim):
    max = np.abs(lim)
    min = -max
    set_lim(min, max)

def show(func, x):
    y = func(x)
    minValue = np.amin(y)
    maxValue = np.amax(y)
    margin = (maxValue - minValue) * 0.1
    set_lim(minValue - margin, maxValue + margin)
    
    plt.plot(x, y)
    plt.show()
    
def show_diff(func, x):
    y = numerical_diff(func, x)
    minValue = np.amin(y)
    maxValue = np.amax(y)
    margin = (maxValue - minValue) * 0.1
    set_lim(minValue - margin, maxValue + margin)
    
    plt.plot(x, y)
    plt.show()

def show_grad_desc(f, init_x, lr = 0.1, step_num = 20):
    x, history = gradient_descent(f, init_x, lr, step_num)
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(history[:,0], history[:, 1], 'o')
    
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
    
    
def del_mod(i):
    del sys.modules["ch" + str(i)]

def gradient_descent(f, init_x, lr = 0.1, step_num = 100):
    x = init_x
    
    history = []
    history.append(x.copy())
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        history.append(x.copy())
    
    return x, np.array(history)