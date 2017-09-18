import numpy as np

def calc_value(a1, a2, x1, x2):
    a = np.array([a1, a2])
    x = np.array([x1, x2])
    return np.sum(a * x)
    
def perceptron(a1, a2, b, x1, x2):
    value = calc_value(a1, a2, x1, x2) + b
    if 0 < value:
        return 1
    else:
        return 0

def AND(x1, x2):
    #value = calc_value(3, 3, x1, x2)
    #theta = 5
    #if theta < value:
    #    return 1
    #else:
    #    return 0
    return perceptron(3, 3, -5, x1, x2)
    
def NAND(x1, x2):
    return perceptron(-3, -3, 5, x1, x2)
    
def OR(x1, x2):
    return perceptron(3, 3, -2, x1, x2)
    
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)