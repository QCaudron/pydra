"""
MDN Outputs module
------------------

Implementation of Mixture Density Networks in Keras.

Summary
-------
Contains all methods associated with the output of the model including sampling
from the model distribution and generating statistics

"""
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import objectives
import numpy as np
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from get_coefficients import get_mixture_coef

def generate_mdn_sample_from_ouput(output, test_size):
    """
    Using the output layer from the prediction on a fitted mdn model
    generate test_size number of samples.

    Parameters
    ----------
        output : array
            layer of neural network ordered mixture weights (unscaled), variance
            (unscaled) and means
        test_size : int
            number of samples to draw from fitted mdn.



    Returns
    ----------
    result : array
        sample from mixture distribution.

    """
    num_components = int(output.shape[1]/3)

    out_mu = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_pi = output[:,2*num_components:]

    result = np.zeros(test_size)
    rn = np.random.randn(test_size)
    mu = 0
    std = 0
    idx = 0

    for i,_ in enumerate(result):
        idx = np.random.choice(num_components, 1, p=out_pi[i])

        mu = out_mu[i,idx]
        std = np.sqrt(out_sigma[i,idx])
        result[i] = mu + rn[i]*std
    return result

def get_stats(output):
    """
    Gets mean and percentile values from output of MDN.

    Parameters
    ----------
        output : array
            final layer of MDN network


    Returns
    ----------
        mixure_mu : array
            means of output

        mixture_sigma : array
            std of output
    """
    num_components = int(output.shape[1]/3)
    pi,sigma,mu = get_mixture_coef(output, num_components=num_components)
    mu = mu.swapaxes(1,0)
    sigma = sigma.swapaxes(1,0)
    pi = pi.swapaxes(1,0)

    mixture_mu = np.sum(pi*mu,axis=0)
    mu_d = np.power((mu - mixture_mu),2.)
    mixture_sigma = np.sqrt(np.sum(pi*(mu_d + sigma**2),axis=0))

    return mixture_mu,mixture_sigma
