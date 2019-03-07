'''
Jonathan Schuba
April 2018
'''


import numpy as np
import csv

class Scaler():
    ''' Scale an input matrix for the appropriate activation function for a NN
    
    For logistic sigmoid:
        Scale data from 0 to 1
        d = (D - m) / (M - m)
        where 
            d = scaled data
            D = input data
            M = max value in that column
            m = min value in that column
    For relu:
        Scale data from 0 to inf
        d = (D - m)
        where 
            d = scaled data
            D = input data
            m = min value in that column
    For standardize:
        Scale data to mean=0 and variance=1
        d = (D - mu)/sigma
        where 
            d = scaled data
            D = input data
            mu = mean value of that column
            sigma = stdev of that column
     
    Parameters
    -------
    activation : string {'relu', 'logistic', 'standardize'}
        The activation function to scale for
        
    Attributes
    --------
    n_components_ : int
        Number of columns that were fitted  
    X_max_ : np array
        Max of each column       
    X_min_ : np array
        Min of each column 
    X_mean_ : np array
        Mean of each column
    X_stdev_ : np array
        Standard Deviation of each column
        
    '''

    def __init__(self, activation = 'relu'):
        self.activation = activation
        
    def fit(self, X):
        ''' Fit the scaler to X '''
    
        X = np.array(X)
        try:
            _, self.n_components_ = np.shape(X)
        except:
            self.n_components_ = 1
        self.X_max_ = np.max(X, axis=0)
        self.X_min_ = np.min(X, axis=0)
        self.X_mean_ = np.mean(X, axis=0)
        self.X_stdev_ = np.std(X, axis=0)
        
    def transform(self, X):
        ''' Transform X
        Return X: transformed np matrix'''
        
        X = np.array(X)
        try:
            m, n = X.shape
        except:
            (m,) = np.shape(X)
            n = 1
        if n != self.n_components_:
            raise ValueError(f"Input matrix columns ({n}) is not same as fitted matrix columns ({self.n_components})")
        
        if self.activation == 'relu':
            X -= self.X_min_

        if self.activation == 'logistic' or self.activation == 'sigmoid':
            X -= self.X_min_
            X /= (self.X_max_ - self.X_min_)

        if self.activation == 'standardize':
            X -= self.X_mean_
            X /= self.X_stdev_
            mean_2 = X.mean(axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
            if not np.allclose(mean_2, 0):
                warnings.warn("Numerical issues were encountered "
                              "when scaling the data "
                              "and might not be solved. The standard "
                              "deviation of the data is probably "
                              "very close to 0. ")
#                X -= mean_2

        return X
    
    def inv_transform(self, X):
        ''' inverse transform X
        Return X: inversely transformed np matrix'''
        
        X = np.array(X)
        try:
            m, n = X.shape
        except:
            (m,) = np.shape(X)
            n = 1
        if n != self.n_components_:
            raise ValueError(f"Input matrix columns ({n}) is not same as fitted matrix columns ({self.n_components})")
        
        if self.activation == 'relu':
            X += self.X_min_
        if self.activation == 'logistic'  or self.activation == 'sigmoid':
            X *= (self.X_max_ - self.X_min_) 
            X += self.X_min_
        if self.activation == 'standardize':
            X *= self.X_stdev_ 
            X += self.X_mean_               
        return X
    
    
