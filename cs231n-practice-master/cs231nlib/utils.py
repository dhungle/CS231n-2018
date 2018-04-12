'''
Created on Jan 15, 2015

@author: dgyHome
@note: this module contains some utilities functions.
'''

import os;
import cPickle as pickle;
import numpy as np;
import matplotlib.pyplot as plt;

def load_CIFAR_batch(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    with open(filename, 'r') as f:
        datadict=pickle.load(f);
        
        X=datadict['data'];
        Y=datadict['labels'];
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
        Y=np.array(Y);
        
        return X, Y;
        
        
def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param ROOT: string of data folder
    @return: Xtr, Ytr: training data and labels
    @return: Xte, Yte: testing data and labels
    """
    
    xs=[];
    ys=[];
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ));
        X, Y=load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
        
    Xtr=np.concatenate(xs);
    Ytr=np.concatenate(ys);
    
    del X, Y;
    
    Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    
    return Xtr, Ytr, Xte, Yte;

def visualize_CIFAR(X_train,
                    y_train,
                    samples_per_class):
    """
    A visualize function for CIFAR 
    """
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    num_classes=len(classes);
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    
    plt.show();
    
def time_function(f, *args):
    """
    Calculate time cost of a function
    
    @param f: a function
    @param *args: respective parameters
    
    @return: total time the function costs 
    """
    
    import time;
    tic=time.time();
    f(*args);
    toc=time.time();
    
    return toc-tic;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
