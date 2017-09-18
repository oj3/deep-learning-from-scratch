import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch4 import numerical_gradient
from layers import *
from optimizer import *
from collections import OrderedDict
from dataset.mnist import load_mnist
 