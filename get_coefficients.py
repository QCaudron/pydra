def get_mixture_coef(output, num_components=24):
    """
    Gets output (layer of NN) and converts into proportion, variance and mean
    for misture density.


    Parameters
    ----------
        output : array
            layer of neural network ordered mixture weights (unscaled), variance
            (unscaled) and means
        num_components : int
            number of mixtures

    Returns
    ----------
    out_pi : array
        weights of mixtures.
    out_sigma : array
        variance of mixtures.
    out_mu : array
        mean of mixtures.

    """

    out_mu = output[:, :num_components]
    out_sigma = output[:, num_components:2*num_components]
    out_pi = output[:, 2*num_components:]
    return out_pi, out_sigma, out_mu
