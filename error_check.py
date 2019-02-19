import warnings

import numpy as np


# List of available distributions
distribution_list = ["Normal", "Beta", "Gamma", "Poisson", "Binomial"]


def check_training_output_values(outputs, distributions, params):
    """
    Checks output values (y) match up with number of defined distributions
    and values are not too small if using beta or gamma mixture models.
    Small values can lead to nans in training as numbers appear outside of
    support since gamma and beta defined on $(0, \\infty)$.
    """

    def check_output_support(output, distribution, eps=1e-3, i=None):
        """
        Check is any output values below eps if using gamma or beta. Raise
        a warning that this may lead to NaNs in training.
        """

        # Output for gammas must be positive
        if np.any(output < 0) and distribution in ["Gamma"]:
            raise ValueError("Can't have output less than or equal to zero with Gamma distribution.")

        # Outputs for betas must been between 0 and 1
        if np.any((output > 1) | (output < 0)) and distribution in ["Beta"]:
            raise ValueError("Can't have output less than zero or greater than one for output with Beta distribution.")

        # Outputs cannot be negative for Poisson and binomial distributions
        if np.any((output < 0)) and distribution in ["Poisson", "Binomial"]:
            raise ValueError("Can't have output less than zero with {} distribution.".format(distribution))

        # Binomial distributions cannot result in anything greater than their parameter n
        if np.any(output > params["binomial_n"]) and distribution in ["Binomial"]:
            raise ValueError("Can't have output greater than binomial_n with Beta distribution.")

        # Values very near one may cause convergence errors in Beta distributions
        if np.any(output > (1-eps)) and distribution in ["Beta"]:
            error_msg = "{}% of values for output {} are between {} and {}. "
            error_msg += "With a Beta distribution as output, this may lead to NaNs in training."
            warnings.warn(error_msg.format(100 * np.mean(output > (1-eps)), i, 1 - eps, 1))

        # Values very near zero may cause problems in Beta or Gamma distributions
        if np.any(output < eps) and distribution in ["Gamma", "Beta"]:
            error_msg = "{}% of values for output {} below {}. "
            error_msg += "With a {} distribution as output, this may lead to NaNs in training."
            warnings.warn(error_msg.format(100 * np.mean(output < eps), i, eps, distribution))

    # In the case of several outputs, check each one
    if isinstance(outputs, list):

        # There must one output per distribution
        if len(outputs) != len(distributions):
            raise ValueError("Size of outputs does not match size of named distributions.")

        # If lengths are valid, check the output support for all distributions
        else:
            for i, (output, distribution) in enumerate(zip(outputs, distributions)):
                check_output_support(output, distribution, i=i)

    # If there's only one output, validate support against that distribution
    else:
        check_output_support(outputs, distributions[0])


def check_distribution(distribution):
    """
    Check if distribution matches one on list.
    """

    if distribution not in distribution_list:
        raise ValueError("Output needs to be of type: {}".format(distribution_list))


def check_output_distributions(output_distributions):
    """
    Check if output distribution list contains only those in distribution_list.
    """

    err_list = [output not in distribution_list for output in output_distributions]
    if np.any(err_list):
        raise ValueError("Output needs to be of type: {}".format(distribution_list))


def check_output_distributions_equals_output_size(output_size, output_distributions):
    """
    Check output distribution length same as output size
    """

    if output_size != len(output_distributions):
        raise ValueError("Output size needs to be same as length of output_distributions.")


def check_binomial_n_defined_if_binomial(params, distributions):
    """
    Check that binomial_n is defined in params if using binomial
    """

    if "Binomial" in distributions:
        if isinstance(params, dict):
            if "binomial_n" not in params:
                raise ValueError("binomial_n needs to be defined in params.")
            elif not isinstance(params["binomial_n"], float):
                raise ValueError("binomial_n must be float.")

        else:
            raise ValueError("params needs to be defined as a dictionary.")
