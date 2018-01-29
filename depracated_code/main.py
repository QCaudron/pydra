from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from mdn import *
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer

def generate(output, testSize, num_components=24, output_dim=1, M=1):
	out_pi = output[:,:num_components]
	out_sigma = output[:,num_components:2*num_components]
	out_mu = output[:,2*num_components:]
	out_mu = np.reshape(out_mu, [-1, num_components, output_dim])
	out_mu = np.transpose(out_mu, [1,0,2])
	# use softmax to normalize pi into prob distribution
	max_pi = np.amax(out_pi, 1, keepdims=True)
	out_pi = out_pi - max_pi
	out_pi = np.exp(out_pi)
	normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
	out_pi = normalize_pi * out_pi
	# use exponential to make sure sigma is positive
	out_sigma = np.exp(out_sigma)
	result = np.random.rand(testSize, M, output_dim)
	rn = np.random.randn(testSize, M)
	mu = 0
	std = 0
	idx = 0
	for j in range(0, M):
		for i in range(0, testSize):
		  for d in range(0, output_dim):
		    idx = np.random.choice(24, 1, p=out_pi[i])
		    mu = out_mu[idx,i,d]
		    std = out_sigma[i, idx]
		    result[i, j, d] = mu + rn[i, j]*std
	return result

def keras_generate(output, testSize, num_components=24):
    print(testSize)
    out_mu = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_pi = output[:,2*num_components:]
    # use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    result = np.zeros(testSize)
    rn = np.random.randn(testSize)
    mu = 0
    std = 0
    idx = 0

    for i,_ in enumerate(result):
        idx = np.random.choice(num_components, 1, p=out_pi[i])
        
        mu = out_mu[i,idx]
        std = np.sqrt(out_sigma[i,idx])
        result[i] = mu + rn[i]*std
    return result

def oneDim2OneDim(epochs=1000,sample_size=1000):
    num_components=24
    output_dim=1
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sample_size))).T
    r_data = np.float32(np.random.normal(size=(sample_size,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)
    #invert training data
    temp_data = x_data
    x_data = y_data
    y_data = temp_data

    model = Sequential()
    model.add(Dense(128,input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(MixtureDensity(output_dim,num_components))

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(),optimizer=opt)
    model.fit(x_data, y_data, batch_size=x_data.size, epochs=epochs, verbose=1)

    x_test = np.float32(np.arange(-15.0,15.0,0.01))
    x_test = x_test.reshape(x_test.size,1)
    y_test = generate(model.predict(x_test), x_test.size)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro',alpha=0.3)
    plt.plot(x_test,y_test[:,:,0],'bo',alpha=0.3)
    plt.show()
    

def oneDim2TwoDim(sample_size=250):
	num_components=24
	output_dim=2

	z_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sample_size))).T
	r_data = np.float32(np.random.normal(size=(sample_size,1)))
	x1_data = np.float32(np.sin(0.75*z_data)*7.0+z_data*0.5+r_data*1.0)
	x2_data = np.float32(np.sin(0.5*z_data)*7.0+z_data*0.5+r_data*1.0)
	x_data = np.dstack((x1_data,x2_data))[:,0,:]

	model = Sequential()
	model.add(Dense(128,input_shape=(1,)))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(MixtureDensity(output_dim,num_components))

	opt = Adam(lr=0.001)
	model.compile(loss=mdn_loss(num_components=24, output_dim=output_dim),optimizer=opt)
	model.fit(z_data, x_data, batch_size=x_data.size, nb_epoch=1000, verbose=1)

	x_test = np.float32(np.arange(-15.0,15.0,0.01))
	x_test = x_test.reshape(x_test.size,1)
	y_test = generate(model.predict(x_test),
					  x_test.size,
					  num_components=num_components,
					  output_dim=output_dim)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(y_test[:,0,0], y_test[:,0,1], x_test, c='r')
	ax.scatter(x1_data, x2_data, z_data, c='b')
	ax.legend()
	plt.show()

def oneDim2OneDim(epochs=1000,sample_size=1000):
    num_components=24
    output_dim=1
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sample_size))).T
    r_data = np.float32(np.random.normal(size=(sample_size,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)
    #invert training data
    temp_data = x_data
    x_data = y_data
    y_data = temp_data

    model = Sequential()
    model.add(Dense(128,input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(MixtureDensity(output_dim,num_components))

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(),optimizer=opt)
    model.fit(x_data, y_data, batch_size=x_data.size, epochs=epochs, verbose=1)

    x_test = np.float32(np.arange(-15.0,15.0,0.01))
    x_test = x_test.reshape(x_test.size,1)
    y_test = generate(model.predict(x_test), x_test.size)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro',alpha=0.3)
    plt.plot(x_test,y_test[:,:,0],'bo',alpha=0.3)
    plt.show()

class MyLayer(Layer):

    def __init__(self, m,v,p, **kwargs):
        self.m = m
        self.v = v
        self.p = p
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        m,v,p = _transform_mixture_coef(self.m,self.v,self.p)
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    def _transform_mixture_coef(m,v,p):
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
        out_p = p
        out_v = v
        out_m = m
        # use softmax to normalize pi into prob distribution
        max_p = K.max(out_p, axis=1, keepdims=True)
        out_p = out_p - max_p
        out_p = K.exp(out_p)
        normalize_p = 1 / K.sum(out_p, axis=1, keepdims=True)
        out_p = normalize_p * out_p
        # use exponential to make sure sigma is positive
        out_v = K.exp(out_v)
        return out_m, out_v, out_p

def variance_transformation(v):
        out_v = v
        out_v = K.exp(out_v)
        return out_v
def proportion_transformation(p):
        out_p = p
        max_p = K.max(out_p, axis=1, keepdims=True)
        out_p = out_p - max_p
        out_p = K.exp(out_p)
        normalize_p = 1 / K.sum(out_p, axis=1, keepdims=True)
        out_p = normalize_p * out_p
        return out_p

def functional_mdn_test(epochs=100,sample_size=1000):
    '''
    Test functional capabilities of keras to build an mdn.
    '''
    cluster_number = 1
    inputs = Input(shape=(1,))
    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    m = Dense(cluster_number, name ='cluster_means')(x)
    v = Dense(cluster_number, name ='cluster_variances')(x)
    v = Lambda(variance_transformation)(v)
    p = Dense(cluster_number, name ='cluster_proportions')(x)
    p = Lambda(proportion_transformation)(p)
    merged_layer = concatenate([m,v,p], axis=-1)
    model = Model(inputs=[inputs], outputs=[merged_layer])
    print(model.summary())

    opt = Adam(lr=0.001)
    model.compile(loss=keras_mdn_loss(num_components=cluster_number),optimizer=opt)

    #construct data
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sample_size))).T
    y_data = np.float32(np.random.normal(loc=x_data))

    model.fit(x_data, y_data, batch_size=200, epochs=epochs, verbose=1)
    
    
    x_test = np.float32(np.arange(-10.0,10.0,0.1))
    x_test = x_test.reshape(x_test.size,1)
    y_test = keras_generate(model.predict(x_test), x_test.size,num_components=cluster_number)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro',alpha=0.3)
    plt.plot(x_test,y_test,'bo',alpha=0.3)
    plt.show()
    
    res = model.predict(x_test)
    y_mu = res[:,0]
    plt.figure(figsize=(8, 8))
    plt.plot(x_test,y_mu,'bo',alpha=0.3)
    plt.show()
    
    
    
if __name__ == '__main__':
    #oneDim2OneDim()
    #y = np.linspace(0,1)
    #print(y.shape)
    #mu = np.transpose(np.ones(15))
    #sigma = np.transpose(np.ones(15))
    #print tf_normal(y[:, np.newaxis], mu, sigma)
    model = functional_mdn_test()
