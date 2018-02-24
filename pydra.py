"""
Keras MDN module
----------------

Implementation of Mixture Density Networks in Keras.

Summary
-------
This is a complete redesign of the mdn module in order to streamline some of the
code, make it more readable and more generalizable to multiple inputs and
outputs.

Routine Listings
----------------
    1. generate_mdn_sample_from_ouput: function
    2. get_mixture_coef: function
    3. tf_normal: function
    4. get_lossfunc: function
    5. mdn_loss: function
    6. variance_transformation: function
    7. proportion_transformation: function


"""

"""
Load libraries
--------------
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
from mdn_outputs import generate_mdn_sample_from_ouput,get_stats
from distributions import tf_normal,tf_gamma
from transformations import variance_transformation,proportion_transformation
from get_coefficients import get_mixture_coef





def get_lossfunc(out_pi, out_sigma, out_mu, y):
    """
    For vector of mixtures with weights out_pi, variance out_sigma and
    mean out_mu (all of shape (,m)) and data y (of shape (n,)) output loss
    function which is the Gaussian mixture negative log-likelihood

    Parameters
    ----------
    out_pi : array (,m)
        weights of mixtures.
    out_sigma : array (,m)
        variance of mixtures.
    out_mu : array (,m)
        mean of mixtures.
    y : array (n,)
        data outputs.

    Returns
    -------
        Negative log-likelihood : float
    """
    #output (n,m)
    result = tf_normal(y, out_mu, out_sigma)
    #output (n,m)
    result = result * out_pi
    #output (n,)
    result = K.sum(result, axis=1, keepdims=True)
    #output (n,)
    result = -K.log(result + 1e-8)
    #output 1
    result = K.mean(result)


    return result

def mdn_loss(num_components=24, output_dim=1):
    """
    Updated version of mdn loss to avoid having to create a custom keras layer.
    Returns loss function (not loss value) for given number of components
    and output dimension.

    Parameters
    ----------
    num_components : int
        number of mixture components
    output_dim : int
        number of output dimensions

    Returns
    -------
        loss : function

    """
    def loss(y, output):
        """
        Loss function.

        Parameters
        ----------
        y : array (n,)
            data
        output : array (,3*m)
            output layer of neural network containing unscaled mixture weights,
            variances and means.

        Returns
        -------
            loss : function

        """
        out_pi, out_sigma, out_mu = get_mixture_coef(output, num_components)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss





def load_mdn_model(cluster_size=10,output_size=1,layers = 3,input_size=1,
                   dense_layer_size=64,print_summary=True):
    """
    Create a keras mixture density model.

    Example
    -------

    model = load_mdn_model()
    model.fit(x, y, batch_size=200, epochs=epochs, verbose=1)

    Parameters
    ----------
    cluster_size : int
        Number of mixture clusters for each output

    output_size : int
        Number of outputs of model

    layers : int
        Number of densely connected layers

    input_size : int
        Dimension of input size

    dense_layer_size : int
        Number of neurons in the densely connected layers

    print_summary : bool
        Choose whether to print summary of constructed MDN
        (useful for debugging).


    Returns
    -------
        keras layer

    """

    def mdn_merged_layer(x,name=None):
        """
        Create merged mdn layer from densely connected layer.
        """
        m = Dense(cluster_size)(x)
        v = Dense(cluster_size)(x)
        v = Lambda(variance_transformation)(v)
        p = Dense(cluster_size)(x)
        p = Lambda(proportion_transformation)(p)
        return concatenate([m,v,p], axis=-1, name=name)


    inputs = Input(shape=(input_size,))

    # Stack densely-connected layers on top of input.
    x = Dense(dense_layer_size, activation='relu')(inputs)
    for _ in range(1,layers):
        x = Dense(dense_layer_size, activation='relu')(x)

    # create multiple mdn merge layers to generate output of model.
    outputs = [mdn_merged_layer(x,name='output_{}'.format(i)) for i in range(output_size)]

    # Instantiate Keras model.
    model = Model(inputs=[inputs], outputs=outputs)
    if print_summary: print(model.summary())

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(num_components=cluster_size),optimizer=opt)

    return model
