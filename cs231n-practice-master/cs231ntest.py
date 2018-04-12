"""
Author : Yuhuang Hu
Date   : 2015-01-07

Perform some tests for the written code
"""

import numpy as np;
import matplotlib.pyplot as plt;

from cs231nlib.classifier import NearestNeighbor;
from cs231nlib.utils import load_CIFAR10;
from cs231nlib.utils import visualize_CIFAR;

## load dataset

Xtr, Ytr, Xte, Yte=load_CIFAR10("CIFAR10");

print Xtr.shape[0];
print Xtr.shape[1];
print Xtr.shape[2];
print Xtr.shape[3];

## plot configuration

plt.rcParams['figure.figsize']=(10.0, 8.0);
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

visualize_CIFAR(X_train=Xtr, y_train=Ytr, samples_per_class=10);



## Testing for Nearest Neighbor Function

nn=NearestNeighbor();