import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions
from keras import backend as K


def gamma(x):
    return K.exp(tf.lgamma(x))


def tf_normal(y, mu, sigma):
    """
    pdf of normal density for n data points and m mixtures.

    Parameters
    ----------
    y : array (n,)
        data
    mu : array (, m)
        mean
    sigma : array (, m)
        variance

    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture
    """

    one_over_root_pi = 1 / tf.sqrt(2 * np.pi)
    result = (y - mu) * (1 / (sigma + 1e-8))
    result = -K.square(result) / 2
    result = K.exp(result) * (1 / (sigma + 1e-8)) * one_over_root_pi

    return result


def tf_gamma(y, alpha, beta):
    """
    pdf of gamma density for n data points and m mixtures.

    Parameters
    ----------
    y : array (n, )
        data
    alpha : array (, m)
        gamma shape parameter
    beta : array (, m)
        gamma shape parameter

    Returns
    -------

    pdf : array (n, m)
        probability for each data point and mixture
    """

    return distributions.Gamma(concentration=alpha, rate=beta).prob(y)


def tf_beta(y, alpha, beta):
    """
    pdf of beta distribution for _n_ data points and _m_ mixtures

    Parameters
    ----------
    y : array (n, )
        data
    alpha : array (, m)
        beta shape parameter
    beta : array (, m)
        beta shape parameter
    Returns
    -------

    pdf : array (n, m)
        probability for each data point and mixture

    """

    return distributions.Beta(alpha, beta).prob(y)


def tf_poisson(y, lbda, _):
    """
    pmf of poisson density for _n_ data points and _m_ mixtures.

    Parameters
    ----------

    y : array (n, )
        data
    lbda : array (, m)
        Poisson rate parameter

    Returns
    -------

    pdf : array (n, m)
        probability for each data point and mixture

    Notes
    -----

    As other distributions take in three parameters, we include a
    redundant third parameter. This would need refactoring.
    """

    return distributions.Poisson(lbda).prob(y)


def tf_binomial(y, n, p):
    """
    pdf of binomial distribution for _n_ data points and _m_ mixtures

    Parameters
    ----------
    y : array (n,)
        data
    n : array (,m)
        n size parameter
    p : array (,m)
        p probability parameter
    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture

    """

    return distributions.Binomial(n, p).prob(y)


def gen_tf_binomial(n):
    """
    generates function for
    pdf of binomial distribution for _n_ data points and _m_ mixtures

    Parameters
    ----------
    n : float32
        n size parameter
    Returns
    -------

    f : function
        Similar form to tf_ functions
    """

    def f(y, p, _):
        return distributions.Binomial(total_count=n, probs=p).prob(y)
    return f
