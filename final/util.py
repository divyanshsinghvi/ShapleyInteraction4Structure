import scipy
import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import norm 
import pandas as pd
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Shapley explanations for models ------

class RegressionGame():
    def __init__(self, X, y = None, function = None, transform = lambda x: x):
        print("Hey")
        self.X = X
        self.F = X.shape[1]
        self.dim = 2**self.F
        self.y = y
        self.function = function
        self.transform = transform

    # helper functions -----
        
    # Use index to generate binary representation of a set
    # Parameters: i = integer
    # Returns: i in binary with width of self.F
    def makeKey(self, i):
        return np.binary_repr(i, self.F)

    # Predict using "full" trained model or function
    # Parameters: X = n x F numpy array
    # Returns: vector of predictions using model trained on all features
    def getWholePrediction(self, X):
        assert self.function is not None
        return self.function(self.transform(X))

    # Predict by masking features not in indexed subset with that from their marginal distributions
    # Parameters: i = integer, x = 1 x F numpy array, X_bg = n x F numpy array
    # Returns: Prediction on X_bg with unmasked features replaced by the value in x
    def getKernelSHAPPrediction(self, i, x, X_bg):
        key = self.makeKey(i)
        S = np.array(list(key), dtype = int)

        replace = (-1 * S + 1).astype(bool)
        X_modified = np.tile(x, (X_bg.shape[0], 1))#.astype(np.float64)
        # print(X_modified)
        X_modified[:,replace] = X_bg[:,replace]
        return np.mean(self.getWholePrediction(X_modified)).reshape((1))

    def getKernelSHAPGame(self, x, X_bg = None):
        if (X_bg is None):
            X_bg = self.X

        predGameMap = map(lambda z: self.getKernelSHAPPrediction(z, x, X_bg).reshape((1)),
                          range(self.dim))
        return np.concatenate(list(predGameMap))


