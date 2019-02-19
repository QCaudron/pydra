import numpy as np
from keras.layers import Input, Dense, Lambda, concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from layers import mdn_layers
from distributions import tf_normal, tf_gamma, tf_beta, tf_poisson, gen_tf_binomial
from transformations import variance_transformation, proportion_transformation, round_transformation
import error_check
import plot_utils


class Pydra(object):
    """
    Main class for constructing Mixture Density Network.

    Example
    -------

    pydra = Pydra()
    pydra.model.fit(x,y,batch_size=200,epochs=epochs,verbose=1)
    """

    @staticmethod
    def instantiate_mdn_model(
        input_size=1, output_size=1, cluster_size=10, layers=3, dense_layer_size=64,
        output_distributions=None, params={"binomial_n": 1000.}, learning_rate=0.001,
        activation="relu", print_summary=True
    ):

        """
        Create a mixture density model in Keras.

        Example
        -------

        model = Pydra.instantiate_mdn_model()
        model.fit(x, y, batch_size=200, epochs=epochs, verbose=1)

        Parameters
        ----------

        input_size : int
            Dimension of input size

        output_size : int
            Number of outputs of model

        cluster_size : int
            Number of mixture clusters for each output

        layers : int
            Number of densely connected layers

        dense_layer_size : int
            Number of neurons in the densely connected layers

        learning_rate: float
            The learning rate for training (uses Adam).

        activation: str
            Activation function for dense layer. Default is "relu".

        print_summary : bool
            Choose whether to print summary of constructed MDN
            (useful for debugging).


        Returns
        -------
            keras layer

        """

        # If output_distributions is undefined, then all output layers are assumed Normal
        if output_distributions is None:
            output_distributions = ["Normal"] * output_size

        # Dictionary for type of output layer depending on distribution
        # ( beta can reuse gamma_merged_layer )
        

        pdfs = {
            "Gamma": tf_gamma,
            "Normal": tf_normal,
            "Beta": tf_beta,
            "Poisson": tf_poisson,
            "Binomial": gen_tf_binomial(params["binomial_n"])
        }

        # Define the input layer
        inputs = Input(shape=(input_size, ))

        # Stack densely-connected layers on top of the input
        x = Dense(dense_layer_size, activation=activation)(inputs)
        for _ in range(1, layers):
            x = Dense(dense_layer_size, activation=activation)(x)

        # Create multiple MDN merge layers to generate the output of model
        outputs = [mdn_layers[dist](x, cluster_size, name="output_{}".format(i)) for i, dist in enumerate(output_distributions)]

        # Instantiate Keras model
        model = Model(inputs=[inputs], outputs=outputs)
        opt = Adam(lr=learning_rate)
        if print_summary:
            print(model.summary())

        # Construct losses
        loss = {
            "output_{}".format(i): mdn_loss(num_components=cluster_size, pdf=pdfs[dist])
            for i, dist in enumerate(output_distributions)
        }

        # Compile the model and return it
        model.compile(loss=loss, optimizer=opt)
        return model

    def __init__(
        self, cluster_size=10, output_size=1, layers=3, input_size=1,
        dense_layer_size=64, print_summary=True, output_distributions="Normal",
        learning_rate=0.001, params={"binomial_n": 1000.}, activation="relu"
    ):

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
            learning_rate: float
                The learning rate for training (uses Adam).
            params: dictionary
                Dictionary of parameters used in output layer.
            activation: str
                Activation function for dense layer. Default is RELU.
        """

        # If the output is a string for distribution type, listify and "broadcast"
        if isinstance(output_distributions, str):
            error_check.check_distribution(output_distributions)
            output_distributions = [output_distributions] * output_size

        # Validate other aspects of distributions
        error_check.check_output_distributions_equals_output_size(output_size, output_distributions)
        error_check.check_output_distributions(output_distributions)
        error_check.check_binomial_n_defined_if_binomial(params, output_distributions)

        # Set model attributes
        self.params = params
        self.output_distributions = output_distributions
        self.model = Pydra.instantiate_mdn_model(
            cluster_size=cluster_size,
            output_size=output_size,
            layers=layers,
            input_size=input_size,
            params=params,
            dense_layer_size=dense_layer_size,
            print_summary=print_summary,
            output_distributions=output_distributions,
            learning_rate=learning_rate,
            activation=activation
        )

    def fit(self, *args, **kwargs):
        """
        Fit the underlying Keras model.
        """

        error_check.check_training_output_values(args[1], self.output_distributions, self.params)

        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict from the MDN model.
        """

        return self.model.predict(*args, **kwargs)

    def sample(self, inputs):
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

        # Generate predictions from the input
        predictions = self.predict(inputs)

        # Generate predictions for all outputs separately
        prediction_samples = []
        for i, (dist, out) in enumerate(zip(self.output_distributions, predictions)):

            # Validate that the distribution seems to be behaving
            error_check.check_distribution(distribution)

            # Grab the means, sigmas, and mixture weights
            out_pi, out_sigma, out_mu = get_mixture_coef(out, self.cluster_size)

            # Initiate an array to hold results
            result = np.zeros(len(out))

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






            samples = sample(out, inputs.size, distribution=dist, params=self.params)
            prediction_samples.append(samples)

        return np.array(prediction_samples).T

    def predict_plot(self, inputs, plot="mean", axis=None):
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

            # If input is multidimensional, grab the right axis
            if (inputs.ndim == 2) and (inputs[1].size > 1):
                if axis is None:
                    axis = 0
                x_test = inputs[:, axis]
            else:
                x_test = inputs

            # Generate predictions from inputs
            output = self.predict(inputs)

            if plot == "mean":
                plot_utils.plot_mean_and_var(
                    output, x_test, distribution=self.output_distributions, params=self.params
                )

            elif plot == "sample":
                raise NotImplementedError("Not yet implemented. Use plot='mean' instead.")
            else:
                raise ValueError("plot should be either 'mean' or 'sample'.")


def get_lossfunc(out_pi, out_sigma, out_mu, y, pdf=tf_normal):
    """
    For vector of mixtures with weights out_pi, variance out_sigma and
    mean out_mu (all of shape (,m)) and data y (of shape (n,)) output loss
    function which is the mixture density negative log-likelihood

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

    # Calculate weighted probs
    result = out_pi * pdf(y, out_mu, out_sigma)

    # Sum
    result = K.sum(result, axis=1, keepdims=True)

    # Take negative log
    result = -K.log(result + 1e-8)

    return K.mean(result)


def mdn_loss(num_components=24, output_dim=1, pdf=tf_normal):
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

        # Get mixture coefficients and shortcut arguments to the loss function
        out_pi, out_sigma, out_mu = get_mixture_coef(output, num_components)
        return get_lossfunc(out_pi, out_sigma, out_mu, y, pdf=pdf)

    return loss


def get_mixture_coef(output, num_components=24):
    """
    Given the output predictions of the mixture density network, returns
    the weights, variances, and means of each mixture cluster.

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
        means of mixtures.

    """

    out_mu = output[:, :num_components]
    out_sigma = output[:, num_components:2*num_components]
    out_pi = output[:, 2*num_components:]
    return out_pi, out_sigma, out_mu

