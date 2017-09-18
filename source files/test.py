import numpy as np

def softmax_test(x):
    if x.ndim == 1:
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    
    x_t = x.T
    exp = np.exp(x_t - np.max(x_t, axis = 0))
    y_t = exp / np.sum(exp, axis = 0)
    return y_t.T