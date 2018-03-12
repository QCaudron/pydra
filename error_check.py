"""
error_checks
------------

List of helper functions to check for errors in input or in other aspects
of the model set up

Summary
-------
Contains all methods associated with error checking.

"""
import numpy as np
import warnings
distribution_list = ['Normal','Beta','Gamma']

def check_training_output_values(outputs,distributions):
    """
    Checks output values (y) match up with number of defined distributions
    and values are not too small if using beta or gamma mixture models (
    small values can lead to nans in training as numbers appear outside of
    support since gamma and beta defined on $(0,\infty)$.
    ).
    """

    def check_output_support(output,distribution,eps=1e-3,i=None):
        """
        Check is any output values below eps if using gamma or beta. Raise
        a warning that this may lead to nans in training.
        """
        if np.any(output<eps) and distribution in ['Gamma','Beta']:
            warnings.warn("{}% of values for output {} below {}. As using the {} distribution for this output, this may lead to nans in training.".\
                          format(np.mean(output<eps),i,eps,distribution)
                          )
        if np.any((output<=0)) and distribution in ['Gamma']:
            raise NameError('Can\'t have output less than or equal to zero with Gamma distribution.')

        if np.any((output>1) | (output<0)) and distribution in ['Beta']:
            raise NameError('Can\'t have output less than zero or greater than one for output with Beta distribution.')



        if np.any(output>1-eps) and distribution in ['Beta']:
            raise UserWarning('{}% of values for output {} between {} and {}. As using the Beta distribution for this output, this may lead to nans in training.'.\
                          format(np.mean(output>1-eps),i,1.-eps,1.)
                          )


    if isinstance(outputs, list):
        if len(outputs) != len(distributions):
            raise NameError('size of outputs does not match size of named distributions')
        else:
            for i,(output,distribution) in enumerate(zip(outputs,distributions)):
                check_output_support(output,distribution,i=i)
    else:
        check_output_support(outputs,distributions[0])


def check_distribution(distribution):
    """
    Check if distribution matches one on list.
    """
    err = 'Output needs to be of type: {}'.format(distribution_list)
    if distribution not in distribution_list:
        raise NameError(err)


def check_output_distributions(output_distributions):
    """
    Check if output distribution list contains only Gamma, Normal or Beta.
    """

    err = 'Output needs to be of type: {}'.format(distribution_list)
    err_list = [True  if o not in distribution_list else False for o in output_distributions ]
    if np.any(err_list):
        raise NameError(err)

def check_output_distributions_equals_output_size(output_size,output_distributions):
    """
    Check output distribution length same as output size
    """
    if output_size != len(output_distributions):
        raise NameError('output size needs to be same as length of output_distributions')
