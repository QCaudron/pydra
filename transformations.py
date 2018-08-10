"""
Transformations
---------------

Transformations for MDN

Summary
-------
Contains all methods associated with the transformation of keras layers
for use in the top MDN layer of the network.

"""
import numpy as np
import math
from keras import backend as K
from keras.layers import Input, Dense, Lambda, concatenate

def variance_transformation(v):
    """
    Transform a layer by exponentiation to convert to the interval $(0,\infty)$.

    Example
    -------

    v = Lambda(variance_transformation)(v)

    Parameters
    ----------
    v : keras layer
        input layer.


    Returns
    -------
        keras layer

    """
    out_v = v
    out_v = K.exp(out_v)
    return out_v


def proportion_transformation(p):
    """
    Transform a layer such that it represents normalised proportions.

    Example
    -------

    v = Lambda(proportion_transformation)(v)

    Parameters
    ----------
    v : keras layer
        input layer.


    Returns
    -------
        keras layer

    """
    out_p = p
    max_p = K.max(out_p, axis=1, keepdims=True)
    out_p = out_p - max_p
    out_p = K.exp(out_p)
    normalize_p = 1 / K.sum(out_p, axis=1, keepdims=True)
    out_p = normalize_p * out_p
    return out_p

def round_transformation(r):
    """
    Transform a layer such that it represents values rounded to between [0,1]

    Example
    -------

    r = Lambda(round_transformation)(r)

    Parameters
    ----------
    r : keras layer
        input layer.


    Returns
    -------
        keras layer

    """
    out_r = .5 + .5*K.tanh(r)
    return out_r
