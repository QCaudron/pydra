"""
gen_data_utils
--------------

Miscellanious functions for generating fake data to test MDN performance.
"""
import numpy as np
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

def u_shape(n=1000,x=None):
    """
    Create upside-down u (unimodal) data

    parameters
    ----------
    n : int
        size of data

    returns
    -------
    x : numpy array
    y : numpy array
    """

    if x is None: x = np.float32(np.random.uniform(0,1,(1, 1000))).T
    y = np.float32(np.random.normal(loc=-10*(x-0.5)*(x-0.5)))
    return x,y

def final_size(n=1000,x=None):
    """
    Simulates a final size like distribution as R0 varies. Includes threshold
    behaviour.

    parameters
    ----------
    n : int
        size of data

    returns
    -------
    x : numpy array
    y : numpy array
    """
    if x is None: x = np.random.uniform(0,1,n);
    ps = x.copy()
    ps[ps<0.5] = 0.
    ps = np.power(ps,0.8)
    mixture = np.random.rand(ps.size) < ps

    m0 = np.random.normal(loc=0.*ps,scale=1.)
    m1 = np.random.normal(loc=10.*ps,scale=2.*ps)
    y = (1-mixture)*m0 + mixture*m1

    x = x = x.reshape(x.size,1)
    return x,y
