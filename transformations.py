from keras import backend as K


def variance_transformation(v):
    """
    Transform a layer by exponentiation to convert to the interval $(0, \\infty)$.

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

    return K.exp(v)


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

    out_p = K.exp(p - K.max(p, axis=1, keepdims=True))
    normalize_p = 1 / K.sum(out_p, axis=1, keepdims=True)
    return normalize_p * out_p


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

    return 0.5 + 0.5 * K.tanh(r)
