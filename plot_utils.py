import pydra
import matplotlib.pyplot as plt

def plot_mean_and_var(output,x_test,distribution=None):
    """
    Plot mean and variance for a given output.
    """
    def mplot(output,distribution,i=None):
        mu,sigma = pydra.get_stats(output,distribution=distribution)
        plt.figure(figsize=(8, 8))
        plt.title('Output {}'.format(i))
        plt.fill_between(x_test.flatten(), mu-sigma, mu+sigma,alpha=0.5)
        plt.plot(x_test.flatten(), mu,linewidth=5.)

    if isinstance(output, list):
        if distribution is None:
            distribution = ['Normal']*len(output)
        for i,dist in zip(range(len(output)),distribution):
            mplot(output[i],dist,i=i)
    else:
        if distribution is None:
            distribution = ['Normal']
                  
        mplot(output,distribution[0])


def plot_samples(output,y,x,x_test,distribution=None):
    """
    Plot samples.
    """
    def mplot(output,y,distribution,i=None):
        y_test = pydra.generate_mdn_sample_from_ouput(output, x_test.size,
        distribution=distribution)
        plt.figure(figsize=(8, 8))
        plt.title('Output {}'.format(i))
        plt.plot(x_test,y_test,'bo',alpha=0.1,label='mdn output')
        plt.plot(x,y,'ro',alpha=0.3,label='data')
        plt.legend();plt.xlabel('x');plt.ylabel('y');

    if isinstance(output, list):
        if distribution is None:
            distribution = ['Normal']*len(output)

        for i,dist in zip(range(len(output)),distribution):
            mplot(output[i],y[i],dist,i=i)
    else:
        if distribution is None:
            distribution = ['Normal']

        mplot(output,y,distribution[0])
