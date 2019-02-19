from keras.layers import Dense, Lambda, concatenate
from transformations import variance_transformation, proportion_transformation, round_transformation


def normal_merged_layer(x, cluster_size, name=None):
    """
    Create merged MDN layer for a normal distribution
    from densely connected layer.
    """

    # Means
    m = Dense(cluster_size)(x)

    # Variances
    v = Dense(cluster_size)(x)
    v = Lambda(variance_transformation)(v)

    # Mixture weights
    p = Dense(cluster_size)(x)
    p = Lambda(proportion_transformation)(p)

    return concatenate([m, v, p], axis=-1, name=name)


def gamma_merged_layer(x, cluster_size, name=None):
    """
    Create merged MDN layer for a gamma distribution
    from densely connected layer.
    """

    alpha = Dense(cluster_size)(x)
    alpha = Lambda(variance_transformation)(alpha)

    beta = Dense(cluster_size)(x)
    beta = Lambda(variance_transformation)(beta)

    p = Dense(cluster_size)(x)
    p = Lambda(proportion_transformation)(p)

    return concatenate([alpha, beta, p], axis=-1, name=name)


def poisson_merged_layer(x, cluster_size, name=None):
    """
    Create merged MDN layer for a Poisson distribution
    from densely connected layer.
    """

    rate = Dense(cluster_size)(x)
    rate = Lambda(variance_transformation)(rate)

    rate_p = Dense(cluster_size)(x)
    rate_p = Lambda(variance_transformation)(rate_p)

    p = Dense(cluster_size)(x)
    p = Lambda(proportion_transformation)(p)

    return concatenate([rate, rate_p, p], axis=-1, name=name)


def binomial_merged_layer(x, cluster_size, name=None):
    """
    Create merged mdn layer for a binomial distribution
    from densely connected layer.
    """

    # We use a round transformation here as ps are independent.
    p = Dense(cluster_size)(x)
    p = Lambda(round_transformation)(p)

    p_ = Dense(cluster_size)(x)
    p_ = Lambda(round_transformation)(p_)

    pi = Dense(cluster_size)(x)
    pi = Lambda(proportion_transformation)(pi)

    return concatenate([p, p_, pi], axis=-1, name=name)


mdn_layers = {
            "Gamma": gamma_merged_layer,
            "Normal": normal_merged_layer,
            "Beta": gamma_merged_layer,
            "Poisson": poisson_merged_layer,
            "Binomial": binomial_merged_layer
        }
