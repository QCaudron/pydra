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
from distributions import tf_normal,tf_gamma,tf_beta
from transformations import variance_transformation,proportion_transformation
from get_coefficients import get_mixture_coef
import error_check as ec
import plot_utils

class Pydra:
    """
    Main class for constructing Mixture Density Network.

    Example
    -------

    pydra = Pydra()
    pydra.model.fit(x,y,batch_size=200,epochs=epochs,verbose=1)
    """
    @staticmethod
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

            return concatenate([alpha,beta,p], axis=-1, name=name)

        # create dictionary for type of output layer depending on distribution
        #beta can re-use gamma_merged_layer
        mlayers = {'Gamma':gamma_merged_layer,'Normal':normal_merged_layer,
                   'Beta':gamma_merged_layer}
        pdfs = {'Gamma':tf_gamma,'Normal':tf_normal,'Beta':tf_beta}

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


        loss_list = [mdn_loss(num_components=cluster_size,pdf=pdfs[dist]) \
                    for dist in output_distributions]

        loss = {'output_{}'.format(i) : loss for i,loss in enumerate(loss_list)}

        model.compile(loss=loss,optimizer=opt)

        return model



    def __init__(self,cluster_size=10,output_size=1,layers = 3,input_size=1,
                 dense_layer_size=64,print_summary=True,
                 output_distributions='Normal'):
        """
        Initialize Pydra class.

        Parameters
        ----------
            cluster_size : int
                number of output clusters
            output_size : int
                dimension of output
            layers : int
                number of densely-connected layers
            input_size : int
                size of inputs
            dense_layer_size : int
                number of neurons in dense layer
            print_summry : bool
                print out summary of network
            output_distributions: list
                list of distribution for outputs. Default is 'Normal'.
        """
        if isinstance(output_distributions, str):
            # if output just a string then turn into array of lenth output_size
            ec.check_distribution(output_distributions)
            output_distributions = [output_distributions]*output_size

        ec.check_output_distributions_equals_output_size(output_size,output_distributions)
        ec.check_output_distributions(output_distributions)
        self.outputs = output_distributions

        self.model = Pydra.load_mdn_model(cluster_size=cluster_size,
        output_size=output_size,layers = layers,input_size=input_size,
        dense_layer_size=dense_layer_size,print_summary=print_summary,
        output_distributions=output_distributions)

        self.predicted_output = None

    def fit(self,*args,**kwargs):
        """
        This is a hack.
        TODO: What we really want is inheritence from the keras model class
        so we get
        all these functions automatically.
        """
        ec.check_training_output_values(args[1],self.outputs)


        return self.model.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        """
        This is a hack.
        TODO: What we really want is inheritence from the keras model class
        so we get
        all these functions automatically.
        """

        output = self.model.predict(*args,**kwargs)
        self.predicted_output = output

        return output

    def generate_mdn_sample_from_ouput(self,inputs):
        """
        Produce samples from fitted model for a given set of inputs.

        Parameters
        ----------
        inputs : numpy array
            inputs into MDN model to predict.

        Returns
        -------
        numpy array
            sample predictions.
        """

        output = self.predict(inputs)
        if isinstance(output, list):
            prediction_samples = []
            for i,dist in zip(range(len(output)),distribution):
                samples = generate_mdn_sample_from_ouput(output[i],
                                                         inputs.size,
                                                         distribution=dist)
                prediction_samples.append(samples)
        else:
            prediction_samples = generate_mdn_sample_from_ouput(output,
                                                         inputs.size,
                                                         distribution=self.outputs[0])

        return prediction_samples

    def predict_plot(self,inputs,plot='mean',axis=None):
            """
            Predict for a set of inputs and then plot.

            Example
            -------

            `
            model = Pydra()
            model.fit(x, y, batch_size=200, epochs=epochs, verbose=1)
            input1 = np.linspace(0,10)
            input2 = val*np.ones(input1.shape)
            input = np.vstack((input1,input2)).T
            model.predict_plot(input,plot='sample')
            `

            Parameters
            ----------
            inputs : numpy array or list
                input used in prediction

            plot : string
                'sample' or 'mean'. Used to determine which type of plot to
                output.

            axis : integer
                Used when model accepts multi-dimensional input. Plots are
                only one-dimensional, so this specifies which input dimension
                to plot over

            Returns
            -------
                None

            """

            # if input is multi-dimensional
            if (inputs.ndim==2) and (inputs[1].size>1):
                axis = 0 if axis is None else axis
                x_test = inputs[:,axis]
            else:
                x_test = inputs

            output = self.predict(inputs)

            if plot=='mean':
                plot_utils.plot_mean_and_var(output,x_test,
                                            distribution=self.outputs)
            elif plot=='sample':
                raise NameError('Not yet implemented. Use plot=\'mean\' instead.')
            else:
                raise NameError('plot either mean or sample.')






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
