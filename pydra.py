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

def check_output_distributions(output_distributions):
    """
    Check if output distribution list contains only Gamma, Normal or Beta.
    """
    dist_list = ['Normal','Beta','Gamma']
    err = 'Output needs to be of type: {}'.format(dist_list)
    err_list = [True  if o not in dist_list else False for o in output_distributions ]
    if np.any(err_list):
        raise NameError(err)

def check_output_distributions_equals_output_size(output_size,output_distributions):
    """
    Check output distribution length same as output size
    """
    if output_size != len(output_distributions):
        raise NameError('output size needs to be same as length of output_distributions')


class Pydra:
    """
    Main class for
    """
    def __init__(self,cluster_size=10,output_size=1,layers = 3,input_size=1,
                 dense_layer_size=64,print_summary=True,
                 output_distributions=['Normal']):

        self.model = load_mdn_model(cluster_size=cluster_size,
        output_size=output_size,layers = layers,input_size=input_size,
        dense_layer_size=dense_layer_size,print_summary=print_summary)

        check_output_distributions_equals_output_size(output_size,output_distributions)
        check_output_distributions(output_distributions)
        self.outputs = output_distributions



def get_lossfunc(out_pi, out_sigma, out_mu, y, pdf=tf_normal):
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
    pdf : function
        defines the probability density funtion for the loss function

    Returns
    -------
        Negative log-likelihood : float
    """
    #output (n,m)
    result = pdf(y, out_mu, out_sigma)
    #output (n,m)
    result = result * out_pi
    #output (n,)
    result = K.sum(result, axis=1, keepdims=True)
    #output (n,)
    result = -K.log(result + 1e-8)
    #output 1
    result = K.mean(result)


    return result

def mdn_loss(num_components=24, output_dim=1,pdf=tf_normal):
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
    pdf : function
        defines the probability density funtion for the loss function

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
        return get_lossfunc(out_pi, out_sigma, out_mu, y, pdf=pdf)
    return loss





def load_mdn_model(cluster_size=10,output_size=1,layers = 3,input_size=1,
                   dense_layer_size=64,print_summary=True,
                   output_distributions=None):
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
    # if output_distributions undefined then all output layers are normal
    if output_distributions is None:
        output_distributions = ['Normal']*output_size

    def normal_merged_layer(x,name=None):
        """
        Create merged mdn layer for a normal distribution
        from densely connected layer.
        """
        m = Dense(cluster_size)(x)

        v = Dense(cluster_size)(x)
        v = Lambda(variance_transformation)(v)

        p = Dense(cluster_size)(x)
        p = Lambda(proportion_transformation)(p)

        return concatenate([m,v,p], axis=-1, name=name)

    def gamma_merged_layer(x,name=None):
        """
        Create merged mdn layer for a gamma distribution
        from densely connected layer.
        """
        alpha = Dense(cluster_size)(x)
        alpha = Lambda(variance_transformation)(alpha)

        beta = Dense(cluster_size)(x)
        beta = Lambda(variance_transformation)(beta)

        p = Dense(cluster_size)(x)
        p = Lambda(proportion_transformation)(p)

        return concatenate([m,v,p], axis=-1, name=name)

    # create dictionary for type of output layer depending on distribution
    mlayers = {'Gamma':gamma_merged_layer,'Normal':normal_merged_layer}
    pdfs = {'Gamma':tf_gamma,'Normal':tf_normal}

    # define input layer
    inputs = Input(shape=(input_size,))

    # Stack densely-connected layers on top of input.
    x = Dense(dense_layer_size, activation='relu')(inputs)
    for _ in range(1,layers):
        x = Dense(dense_layer_size, activation='relu')(x)

    # create multiple mdn merge layers to generate output of model.
    outputs = [mlayers[dist](x,name='output_{}'.format(i)) \
              for i,dist in enumerate(output_distributions)]

    # Instantiate Keras model.
    model = Model(inputs=[inputs], outputs=outputs)
    if print_summary: print(model.summary())

    opt = Adam(lr=0.001)

    # TODO: check this code works for gamma distribution
    loss_list = [mdn_loss(num_components=cluster_size,pdf=pdfs[dist]) \
                for dist in output_distributions]

    loss = {'output_{}'.format(i) : loss for i,loss in enumerate(loss_list)}

    model.compile(loss=loss,optimizer=opt)

    return model
