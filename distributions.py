"""
Distributions
------------------

Implementation of Mixture Density Networks in Keras.

Summary
-------
Contains all methods associated with the output of the model including sampling
from the model distribution and generating statistics

"""
import numpy as np
import math
from keras import backend as K

def tf_normal(y, mu, sigma):
    '''
    pdf of normal density for _n_ data points and _m_ mixtures.

    Parameters
    ----------
    y : array (n,)
        data
    mu : array (,m)
        mean
    sigma : array (,m)
        variance
    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture
    '''
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu #broadcasting converts this to two-dimensional array.
    #result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI

    #result = K.prod(result, axis=[0]) #should be (,m) array

    return result

def tf_gamma(y,alpha,beta):
    '''
    pdf of gamma density for _n_ data points and _m_ mixtures.

    Parameters
    ----------
    y : array (n,)
        data
    alpha : array (,m)
        gamma shape parameter
    beta : array (,m)
        gamma shape parameter
    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture
    '''
    Z = K.gamma(alpha) * beta**alpha
    return y**(alpha - 1) * K.exp(-x * beta) / Z