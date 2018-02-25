# Example Notebooks
----

Contains example notebooks for fitting an MDN model for one to multi-dimensional
input and output. See below for a description of each notebook. All notebooks
were ran in `Python 3.6`.

* [Example 1](one_input_to_one_output_example.ipynb): This notebook contains two
examples of fitting a one-input one-ouput model where the input controls the mean
and formation of different mixtures.
* [Example 2](one_input_to_two_output_example.ipynb): This notebook contains one
example of fitting a one-input to two-output model where the input controls the mean
and formation of different mixtures. This shows off the power of the functional
api for keras.
* [Example 3](two_inputs_to_one_output_example.ipynb): Here we consider more than
one model input to simulate multiple epidemiological parameters giving rise to
a final size distribution.
* [Example 4](Negative_binomial_test.ipynb): This notebook considers fitting
to a negative binomial distribution. This is currently not working as code needs
to be adapted to include a Gamma mixture as an output. 
