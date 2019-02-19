import pydra
import matplotlib.pyplot as plt


def plot_mean_and_var(output, x_test, distribution=None, params=None):
    """
    Plot mean and variance for a given output.
    """

    def plot(output, distribution, i=None):

        # Get moments of the model output
        mu, sigma = pydra.get_stats(output, distribution=distribution, params=params)

        # Plot an area for the standard deviation and a line for the mean
        plt.figure(figsize=(8, 8))
        plt.title("Output {}".format(i))
        plt.fill_between(x_test.flatten(), mu - sigma, mu + sigma, alpha=0.5)
        plt.plot(x_test.flatten(), mu, lw=5)

    # Wrap output in a list if it's not already one, so we can iterate
    if not isinstance(output, list):
        output = [output]

    # If the distribution isn't defined, assume it's Normal
    if distribution is None:
        distribution = ["Normal"] * len(output)

    # Plot each output separately
    for idx, (dist, out) in enumerate(zip(distribution, output)):
        plot(out, dist, i=idx)


def plot_samples(output, y, x, x_test, distribution=None):
    """
    Plot samples.
    """

    def plot(output, distribution, y, i=None):

        # Generate samples from the model
        y_test = pydra.sample_from_output(output, x_test.size, distribution=distribution)

        plt.figure(figsize=(8, 8))
        plt.title("Output {}".format(i))
        plt.plot(x_test, y_test, "bo", alpha=0.1, label="MDN output")
        plt.plot(x, y, "ro", alpha=0.3, label="data")
        plt.legend()

    # Wrap output and y in a list if they not already one, so we can iterate
    if not isinstance(output, list):
        output = [output]
    if not isinstance(y, list):
        y = [y]

    # If the distribution isn't defined, assume it's Normal
    if distribution is None:
        distribution = ["Normal"] * len(output)

    # Plot each output separately
    for idx, (dist, out, y_i) in enumerate(zip(distribution, output, y)):
        plot(out, dist, y_i, i=idx)
