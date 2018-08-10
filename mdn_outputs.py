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
import error_check as ec


def generate_mdn_sample_from_ouput(output, test_size,distribution = 'Normal',
                                   params = None):
    """
    Using the output layer from the prediction on a fitted mdn model
    generate test_size number of samples. (Note output corresponds to a
    one-dimensional output).

    Parameters
    ----------
        output : array
            layer of neural network ordered mixture weights (unscaled), variance
            (unscaled) and means
        test_size : int
            number of samples to draw from fitted mdn.
            deprecated.
        distribution: string
            distribution of output. Can be Normal, Gamma or Beta.



    Returns
    ----------
    result : array
        sample from mixture distribution.

    """
    ec.check_distribution(distribution)

    num_components = int(output.shape[1]/3)

    out_mu = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_pi = output[:,2*num_components:]

    result = np.zeros(output.shape[0])

    mu = 0
    std = 0
    idx = 0

    for i,_ in enumerate(result):
        idx = np.random.choice(num_components, 1, p=out_pi[i])
        if(distribution is 'Normal'):
            mu = out_mu[i,idx]
            std = np.sqrt(out_sigma[i,idx])
            result[i] = mu + np.random.randn()*std
        elif(distribution is 'Gamma'):
            alpha = out_mu[i,idx]
            beta = out_sigma[i,idx]
            result[i] = np.random.gamma(alpha,1/beta)
        elif(distribution is 'Beta'):
            alpha = out_mu[i,idx]
            beta = out_sigma[i,idx]
            result[i] = np.random.beta(alpha,beta)
        elif(distribution is 'Poisson'):
            rate = out_mu[i,idx]
            result[i] = np.random.poisson(rate)
        elif(distribution is 'Binomial'):
            p = out_mu[i,idx]
            n = out_sigma[i,idx]
            result[i] = np.random.binomial(params['binomial_n'],p)
        else:
            raise NameError('{} not a distribution'.format(distribution))
    return result

# TODO Extend to other distributions: beta and gamma
def get_stats(output,distribution = 'Normal',params = None):
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
    ec.check_distribution(distribution)

    num_components = int(output.shape[1]/3)
    pi,sigma,mu = get_mixture_coef(output, num_components=num_components)
    mu = mu.swapaxes(1,0)
    sigma = sigma.swapaxes(1,0)
    pi = pi.swapaxes(1,0)

    if distribution == 'Normal':
        mixture_mu = np.sum(pi*mu,axis=0)
        mu_d = np.power((mu - mixture_mu),2.)
        mixture_sigma = np.sqrt(np.sum(pi*(mu_d + sigma**2),axis=0))
    elif(distribution is 'Gamma'):
        alpha,beta = mu,sigma
        mixture_mu = np.sum(pi*alpha/beta,axis=0)
        mixture_sigma = np.sqrt(np.sum(pi*alpha/(beta**2),axis=0))
    elif(distribution is 'Beta'):
        alpha,beta = mu,sigma
        mixture_mu = np.sum(pi*alpha/(alpha+beta),axis=0)
        v = alpha*beta/(np.power(alpha+beta,2)*(alpha+beta+1))
        mixture_sigma = np.sqrt(np.sum(pi*v,axis=0))
    elif(distribution is 'Poisson'):
        mixture_mu = np.sum(pi*mu,axis=0)
        mixture_var = np.sum(pi*(mu**2 + mu),axis=0) - mixture_mu**2
        mixture_sigma = np.sqrt(mixture_var)
    elif(distribution is 'Binomial'):
        # TODO
        # mu ==n, sigma == p
        p = mu
        n = params['binomial_n']
        mixture_mu = np.sum(pi*n*p,axis=0)
        v = n*p*(1-p)
        mixture_var = np.sum(pi*(n*n*p*p+v),axis=0) - mixture_mu**2
        mixture_sigma = np.sqrt(mixture_var)
    else:
        raise NameError('{} not a distribution'.format(distribution))

    return mixture_mu,mixture_sigma
