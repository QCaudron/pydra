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
    out_mu = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_pi = output[:,2*num_components:]
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    '''
    pdf of normal density for _n_ data points and _m_ mixtures.

    Parameters
    ----------
    y : array (n,)
        data
    mu : array (,m)
        mean
    sigma : array (,m)
        variance
    Returns
    -------

    pdf : array (n,m)
        probability for each data point and mixture
    '''
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu #broadcasting converts this to two-dimensional array.
    #result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI

    #result = K.prod(result, axis=[0]) #should be (,m) array

    return result

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
    print(model.summary())

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(num_components=cluster_size),optimizer=opt)

    return model
