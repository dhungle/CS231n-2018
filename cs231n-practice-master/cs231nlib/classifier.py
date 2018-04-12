'''
Created on Jan 7, 2015

@author: Yuhuang Hu
@note: This module consists of several classifiers class.
'''

import numpy as np;

class NearestNeighbor:
    """
    @note: this class provides an implementation of NearestNeighbor classifier.
    """
    
    def __init__(self):
        pass
    
    def train(self, x, y):
        """
        @param x: N x D matrix, each row is a D-dimensional vector.
        @param y: N x 1 vector, describes corresponding labels. 
        """
        
        self.Xtr=x;
        self.Ytr=y;

    def predict(self, x):
        """
        @param x: N x D matrix, each row is a D-dimensional vector.
        @return: a 1D vector that consists of all predicted labels 
        """
        
        num_test=x.shape[0];
        
        Y_pred=np.zeros(num_test, dtype=self.Ytr.dtype);
        
        for i in xrange(num_test):
            
            distances=np.sum(np.abs(self.Xtr-x[i,:]), axis=1);
            min_index=np.argmax(distances);
            Y_pred[i]=self.Ytr[min_index];
            
        return Y_pred;
        
        
class KNearestNeighbor:
    """
    Implementation of K Nearest Neighbor (KNN) classifier
    
    The code is adapted from C231n assignment kit.
    """
    def __init__(self):
        pass;
    
    def train(self, X, y):
        """
        @param x: N x D matrix, each row is a D-dimensional vector.
        @param y: N x 1 vector, describes corresponding labels.  
        """
        
        self.X_train=X;
        self.Y_train=y;
        
    def predict(self, X, k=1):
        """
        @param X: N x D matrix, each row is a D-dimensional vector.
        @param k: number of voting nearest neighbors
        @return: a 1D vector that consists of all predicted labels
        """
        
        dists=self.compute_distance(X);
        
        return self.predict_labels(dists=dists, k=k);
        
    def compute_distance(self, X):
        """
        @param X: M x D matrix, each row is a test point.
        @return: dists: distance matrix 
        """
        
        num_test=X.shape[0];
        num_train=self.X_train.shape[0];
        
        dists=np.zeros((num_test, num_train));
        
        # calculate distance 
        
        return dists;
        
    def predict_labels(self, dists, k=1):
        """
        @param dists: distance matrix
        @param k: number of voting nearest neighbors
        @return y: 1-D vector that contains predicted label  
        """
        
        num_test=dists.shape[0];
        y_pred=np.zeros(num_test);
        
        for i in xrange(num_test):
            cloest_y=[];
            
        return y_pred;
        