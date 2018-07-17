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
import tensorflow as tf
import tensorflow.contrib.distributions as dist

def gamma(x):
    return K.exp(tf.lgamma(x))

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
    #Z = gamma(alpha) * K.pow(beta,alpha)
    #return K.pow(y,(alpha - 1)) * K.exp(-y * beta) / Z
    return dist.Gamma(concentration=alpha, rate=beta).prob(y)

def tf_beta(y,alpha,beta):
    '''
    pdf of beta distribution for _n_ data points and _m_ mixtures

    Parameters
    ----------
    y : array (n,)
        data
    alpha : array (,m)
        beta shape parameter
    beta : array (,m)
        beta shape parameter
    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture

    '''
    #Z = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    #return y**(alpha - 1) * (1 - y)**(beta - 1) / Z
    return dist.Beta(alpha,beta).prob(y)

def tf_poisson(y,lbda,_):
    '''
    pmf of poisson density for _n_ data points and _m_ mixtures.

    Parameters
    ----------

    y : array (n,)
        data
    lbda : array (,m)
        Poisson rate parameter

    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture

    Notes
    -----

    As other distributions take in three parameters we include a redundant third
    parameter. This would need refactoring.
    '''
    #Z = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    #return y**(alpha - 1) * (1 - y)**(beta - 1) / Z
    return dist.Poisson(lbda).prob(y)
