"""
Ideas
-----

1. re-write output layer cleanly for just one dimension.
Then if data is more than one-dimension, treat each dimension independently
as a one-dimensional mixture. The neural network should be able to work
out the relationship between the dimensions.

2. Instead of input as one layer have input as three seperate layers. One for
the weights, one for the means and one for the variances. We would then just
need to re-write the loss function. This would make it a lot simpler to read
and alter.
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math

def get_and_transform_mixture_coef(output, num_components=24, output_dim=1):
    """
    Only called by mdn_loss. Gets output (layer of NN) and softmax
    transforms the mixture weights and re-scales variance weights such that they
    are positive.

    Parameters
    ----------
        output : array
            layer of neural network ordered mixture weights (unscaled), variance
            (unscaled) and means
        num_components : int
            number of mixtures
        output_dim : int
            output dimension

    Returns
    ----------
    out_pi : array
        weights of mixtures.
    out_sigma : array
        variance of mixtures.
    out_mu : array
        mean of mixtures.

    """
    out_pi = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_mu = output[:,2*num_components:]
    out_mu = K.reshape(out_mu, [-1, num_components, output_dim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu

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
    print(result.get_shape())
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
        out_pi, out_sigma, out_mu = get_and_transform_mixture_coef(output, num_components, output_dim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss

def keras_mdn_loss(num_components=24, output_dim=1):
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

class MixtureDensity(Layer):
    """
    Keras output layer for a mixture density network.

    Attributes
    ----------
    kernel_dim : int
        Number of kernels

    num_components : int
        Number of mixtures

    Methods
    -------
    build(self, input_shape)
        Initilalize layer for given input shape (previous layer of network).
    call(self, x, mask=None)
        Call layer.
    get_output_shape_for(self, input_shape)
        Get output shape for given input shape and number of components and
        number of dimensions.

    Examples
    --------
    Called as MixtureDensity(output_dim,num_components) e.g.

    ```python
    model = Sequential()
    model.add(Dense(128,input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(MixtureDensity(output_dim,num_components))
    ```
    """
    def __init__(self, kernel_dim, num_components, **kwargs):
        """
        Loss function.

        Parameters
        ----------
        kernel_dim : int
            dimension of output. Kernel dimension is a very bad name for this.
        num_components : int
            Number of mixtures

        Returns
        -------
        MixtureDensity class instance

        """
        #TODO: Ability to change number of hidden dimensions in this final layer
        self.hidden_dim = 24
        # specify dimension of model output
        self.kernel_dim = kernel_dim
        # specifiy number of mixtures
        self.num_components = num_components
        #call __init__ function on super class (keras layer class).
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build mixture density layer with output dimension 3 x number of mixtures
        and hidden layer set by self.hidden_dim

        Parameters
        ----------
        input_shape : array


        Returns
        -------
        None

        """
        self.input_dim = input_shape[1]
        #TODO: See comment below.
        """
        2 + output dimension. ie. for one-dimension this is 3*(no. of mixtures)
        and two-dimensional data this is 4*(no. of mixtures). The reason being
        that means only are different for different dimensions variances are
        shared (which is a limitation) and weights are shared (necessarily).

        This would need extending or correcting.
        """
        self.output_dim = self.num_components * (2+self.kernel_dim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.input_dim, self.hidden_dim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hidden_dim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hidden_dim, self.output_dim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.output_dim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        """
        Call layer. Takes weights passing them through tanh sctivation and
        returns output.
        """
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
