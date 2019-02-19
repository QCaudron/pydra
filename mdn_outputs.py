import numpy as np
from get_coefficients import get_mixture_coef
import error_check


def sample(output, n_samples, distribution="Normal", params=None):
    """
    Using the output layer from the prediction on a fitted MDN model,
    generate n_samples number of samples.
    (output corresponds to a one-dimensional output).

    Parameters
    ----------
        output : array
            layer of neural network ordered mixture weights (unscaled), variance
            (unscaled) and means
        n_samples : int
            number of samples to draw from fitted mdn.
            deprecated.
        distribution: string
            distribution of output. Can be Normal, Gamma or Beta.

    Returns
    ----------
    result : array
        sample from mixture distribution.

    """

    # Validate that all distributions seem to be behaving
    error_check.check_distribution(distribution)

    # Count components in the mixture model
    num_components = int(output.shape[1] / 3)

    # Grab the means, sigmas, and mixture weights
    out_mu = output[:, :num_components]
    out_sigma = output[:, num_components:2*num_components]
    out_pi = output[:, 2*num_components:]

    # Initiate an array to hold results
    result = np.zeros(output.shape[0])

    mu = 0
    std = 0
    idx = 0

    # For each component in the mixture
    for i in range(len(result)):

        # Grab one mixture at random, weighted by its mixture weight
        idx = np.random.choice(num_components, 1, p=out_pi[i])

        # Generate random numbers per the distribution type for this component
        if distribution == "Normal":
            mu = out_mu[i, idx]
            std = np.sqrt(out_sigma[i, idx])
            result[i] = mu + std * np.random.randn()

        elif distribution == "Gamma":
            alpha = out_mu[i, idx]
            beta = out_sigma[i, idx]
            result[i] = np.random.gamma(alpha, 1 / beta)

        elif distribution == "Beta":
            alpha = out_mu[i, idx]
            beta = out_sigma[i, idx]
            result[i] = np.random.beta(alpha, beta)

        elif distribution == "Poisson":
            rate = out_mu[i, idx]
            result[i] = np.random.poisson(rate)

        elif distribution == "Binomial":
            p = out_mu[i, idx]
            result[i] = np.random.binomial(params['binomial_n'], p)

        else:
            raise ValueError("{} not a valid distribution.".format(distribution))

    return result


def get_stats(output, distribution="Normal", params=None):
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

    # Validate that all distributions seem to be behaving
    error_check.check_distribution(distribution)

    # Get number of components
    num_components = int(output.shape[1] / 3)

    # Get the mixture coefficients
    pi, sigma, mu = get_mixture_coef(output, num_components=num_components)
    mu = mu.swapaxes(1, 0)
    sigma = sigma.swapaxes(1, 0)
    pi = pi.swapaxes(1, 0)

    if distribution == "Normal":
        mixture_mu = np.sum(pi * mu, axis=0)
        mu_d = (mu - mixture_mu) ** 2
        mixture_sigma = np.sqrt(np.sum(pi * (mu_d + sigma**2), axis=0))

    elif distribution == "Gamma":
        alpha, beta = mu, sigma
        mixture_mu = np.sum(pi * alpha / beta, axis=0)
        mixture_sigma = np.sqrt(np.sum(pi * alpha / (beta**2), axis=0))

    elif distribution == "Beta":
        alpha, beta = mu, sigma
        mixture_mu = np.sum(pi * alpha / (alpha + beta), axis=0)
        v = alpha * beta / (((alpha + beta)**2) * (alpha + beta + 1))
        mixture_sigma = np.sqrt(np.sum(pi * v, axis=0))

    elif distribution == "Poisson":
        mixture_mu = np.sum(pi * mu, axis=0)
        mixture_var = np.sum(pi * (mu**2 + mu), axis=0) - mixture_mu ** 2
        mixture_sigma = np.sqrt(mixture_var)

    elif distribution == "Binomial":
        # TODO
        # mu ==n, sigma == p
        p = mu
        n = params['binomial_n']
        mixture_mu = np.sum(pi * n * p, axis=0)
        v = n * p * (1 - p)
        mixture_var = np.sum(pi * (n * n * p * p + v), axis=0) - mixture_mu ** 2
        mixture_sigma = np.sqrt(mixture_var)

    else:
        raise ValueError("{} not a valid distribution.".format(distribution))

    return mixture_mu, mixture_sigma
