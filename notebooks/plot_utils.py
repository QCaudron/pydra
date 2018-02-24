import sys
sys.path.append('../')
import pydra
import matplotlib.pyplot as plt

def plot_mean_and_var(output,x_test):
    def mplot(output,i=None):
        mu,sigma = pydra.get_stats(output)
        plt.figure(figsize=(8, 8))
        plt.title('Output {}'.format(i))
        plt.fill_between(x_test.flatten(), mu-sigma, mu+sigma,alpha=0.5)
        plt.plot(x_test.flatten(), mu,linewidth=5.)

    if isinstance(output, list):
        for i in range(len(output)):
            mplot(output[i],i=i)
    else:
        mplot(output)


def plot_samples(output,y,x,x_test):
    def mplot(output,y,i=None):
        y_test = pydra.generate_mdn_sample_from_ouput(output, x_test.size)
        plt.figure(figsize=(8, 8))
        plt.title('Output {}'.format(i))
        plt.plot(x_test,y_test,'bo',alpha=0.1,label='mdn output')
        plt.plot(x,y,'ro',alpha=0.3,label='data')
        plt.legend();plt.xlabel('x');plt.ylabel('y');

    if isinstance(output, list):
        for i in range(len(output)):
            mplot(output[i],y[i],i=i)
    else:
        mplot(output,y)
