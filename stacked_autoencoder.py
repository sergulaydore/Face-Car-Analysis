import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn import datasets, linear_model, cross_validation, grid_search
import numpy
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn import decomposition
# def load_data(dataset):
#     ''' Loads the dataset

#     :type dataset: string
#     :param dataset: the path to the dataset (here MNIST)
#     '''

def gradient_updates_momentum(cost, params, learning_rate, momentum=0.9):

    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0.)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates    
    


def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    # data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # train_set_x, train_set_y = shared_dataset(train_set)

    # rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
    #         (test_set_x, test_set_y)]
    # return rval

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]     

class dA(object):
    def __init__(self, numpy_rng, n_visible, n_hidden, theano_rng = None, input = None, W = None, bhid = None,bvis = None ):
                     
         self.n_visible = n_visible
         self.n_hidden = n_hidden
         
         # create a Theano random generator that gives symbolic random values
         if not theano_rng:
             theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
             
         if not W:
             initial_W = np.asarray(
             numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                               high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                               size = (n_visible, n_hidden)),
                                dtype = theano.config.floatX)
                                
             W = theano.shared(value = initial_W, name='W', borrow=True)  
              
         if not bvis:
             bvis = theano.shared( value = np.zeros(
                                           n_visible, dtype = theano.config.floatX 
                                           ),
                                    borrow = True
                                 )
                                 
         if not bhid:
             bhid = theano.shared( value = np.zeros(
                                           n_hidden, dtype = theano.config.floatX 
                                           ),
                                    name = 'b',       
                                    borrow = True
                                 )
         self.W = W
         self.b = bhid
         self.b_prime = bvis
         self.W_prime = self.W.T
         self.theano_rng = theano_rng
         self.x = input
         
         self.params = [self.W, self.b, self.b_prime]
    def get_corrupted_input(self, input, corruption_level):
         """
         Keeps '1-corruption_level' entries of the inputs the same and
         zero-out randomly selected subset of size 'corruption_level'
         """     
         return self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level) * input  
             
    def get_hidden_values(self, input): 
         """
         Computes the values in hidden layer
         """
         return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
             
    def get_reconstructed_input(self, hidden):
         """
         Computes the reconstructed input given the values of the hidden layer
         """
         return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
         
    def get_cost_updates(self, corruption_level, learning_rate):     
         """
         Computes the cost and the updates for one training step pf the dA
         """
         tilde_x = self.get_corrupted_input(self.x, corruption_level)
         y = self.get_hidden_values(tilde_x)
         self.z = self.get_reconstructed_input(y)
                     
         cost = T.mean((self.x-self.z)**2)
         gparams = T.grad(cost, self.params)
         # updates = [(param, param-learning_rate * gparam)
         #             for param, gparam in zip(self.params, gparams)
         #             ]
         updates = gradient_updates_momentum(cost, self.params, learning_rate)            
                     
         return (cost, updates)  

class LogisticRegression(object):
    """Multi-class Logistic Regression Class """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression     """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution."""
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()    


class SdA(object):
	""" Stacked denoising auto-encoder class (SdA) """
	def __init__(self, numpy_rng, theano_rng = None, n_ins = 110, hidden_layers_sizes = [80, 30, 10], 
				n_outs = 2, corruption_levels = [0.1, 0.1]):

		self.sigmoid_layers = []
		self.dA_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)

		assert self.n_layers > 0

		if not theano_rng:
			theano_rng = RandomStreams (numpy_rng.randint(2**30))

		# allocate symbolic variables for the data
		self.x = T.matrix('x')
		self.y = T.ivector('y')

		for i in xrange(self.n_layers):
			# construct the sigmoidal layer
			# the size of the input is either the number of hidden units
			# or the layer below or the input size if we are on the first layer
			if i == 0:
				input_size = n_ins
			else:
				input_size = hidden_layers_sizes[i-1]

			if i == 0:
				layer_input = self.x
			else:
				layer_input = self.sigmoid_layers[-1].output

			sigmoid_layer = HiddenLayer(rng = numpy_rng, input = layer_input, n_in = input_size,
										n_out = hidden_layers_sizes[i], activation = T.nnet.sigmoid)
			# add the layer to our list of layers
			self.sigmoid_layers.append(sigmoid_layer)
			self.params.extend(sigmoid_layer.params)

			# Construct a denoising autoencoder that shared weights with this layer
			dA_layer = dA(numpy_rng = numpy_rng, theano_rng = theano_rng, input = layer_input, n_visible = input_size,
							n_hidden = hidden_layers_sizes[i], W = sigmoid_layer.W, bhid = sigmoid_layer.b)

			self.dA_layers.append(dA_layer)

		self.logLayer = LogisticRegression(input = self.sigmoid_layers[-1].output,
			                                n_in = hidden_layers_sizes[-1], n_out = n_outs)
		self.params.extend(self.logLayer.params)
		# compute the cost for second phase of training
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.errors(self.y)

	def pretraining_functions(self, train_set_x, batch_size):
		index = T.lscalar('index') # index to a minibatch
		corruption_level = T.scalar('corruption') # % of corruotion to use
		learning_rate = T.scalar('lr') # learning rate to use
		batch_begin = index * batch_size
		batch_end = batch_begin + batch_size

		pretrain_fns = []
		for dA in self.dA_layers:
			cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
			fn = theano.function( inputs = [index, theano.Param(corruption_level, default = 0.2),
				 									theano.Param(learning_rate, default = 0.1)],
				 					outputs = cost, updates = updates, givens = {self.x: train_set_x[batch_begin: batch_end]}
				 					)
			pretrain_fns.append(fn)

		return pretrain_fns

	def build_finetune_functions(self, datasets, batch_size, learning_rate):


		(train_set_x, train_set_y) = datasets[0]
		(valid_set_x, valid_set_y) = datasets[1]
		(test_set_x, test_set_y) = datasets[2]

		# compute number of minibatches for training, validatio and testing
		n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]
		n_valid_batches /= batch_size
		n_test_batches = test_set_x.get_value(borrow = True).shape[0]
		n_test_batches /= batch_size

		index = T.lscalar('index') # index to a minibatch

		# compute the gradients with respect to model parameters
		gparams = T.grad(self.finetune_cost, self.params)

		# compute list of fine-tuning updates
		# updates = [ (param, param - gparam * learning_rate) 
		# 			 for param, gparam in zip(self.params, gparams)
		# 			 ]

		updates = gradient_updates_momentum(self.finetune_cost, self.params, learning_rate) 



		train_fn = theano.function( inputs = [index], outputs = self.finetune_cost, updates = updates,
									givens = { self.x: train_set_x[ index * batch_size: (index + 1) * batch_size],
												self.y: train_set_y[ index * batch_size: (index + 1) * batch_size ]}, 
									name = 'train')

		test_score = theano.function( inputs = [], outputs = self.errors,
										givens={self.x: test_set_x, self.y: test_set_y}, name = 'test' )

		valid_score = theano.function( inputs = [ ], outputs = self.errors,
			givens={self.x: valid_set_x, self.y: valid_set_y}, name = 'valid' )
		# create a function that scans the entire validation set
		# def valid_score():
		# 	return [valid_score_i(valid_set_x, valid_set_y)]

		# # create a function that scans the entire test set
		# def test_score():
		# 	return [test_score_i(test_set_x, test_set_y)]

		return train_fn, valid_score(), test_score()

from data_prep import *

from sklearn.utils import shuffle
X_eeg, y_eeg = shuffle(X_eeg, y_eeg, random_state=0)
pca = decomposition.PCA(n_components=None, copy=True, whiten=True)
pca.fit(X_eeg)
X_eeg = pca.transform(X_eeg)

finetune_lr=0.08; pretraining_epochs=10; pretrain_lr=0.08; training_epochs=1000; batch_size=10

X_train_valid, test_set_x, y_train_valid, test_set_y = cross_validation.train_test_split( X_eeg, y_eeg,
																				 test_size=0.1, random_state=0)
train_set_x, valid_set_x, train_set_y, valid_set_y = cross_validation.train_test_split( X_train_valid, y_train_valid,
																						 test_size=0.2, random_state=0)
test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)

datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

n_train_batches = train_set_x.get_value(borrow = True).shape[0]
n_train_batches /= batch_size

numpy_rng = numpy.random.RandomState(89677)
print '... building the model'

sda = SdA( numpy_rng = numpy_rng)
print '... getting the pretraining functions'
pretrain_fns = sda.pretraining_functions(train_set_x = train_set_x, batch_size = batch_size)
start_time = time.clock()
corruption_levels = [.1, .2, .3]
c_all = np.zeros((sda.n_layers,pretraining_epochs ))
for i in xrange(sda.n_layers):
	# go through pretraining epochs
	for epoch in xrange(pretraining_epochs):
		# go through the training set
		c = []
		# if epoch < 50:
		# 	pretrain_lr = 0.05
		# elif 50 <= epoch < 100:
		# 	pretrain_lr = 0.4
		# elif 100 <= epoch < 200:
		# 	pretrain_lr = 0.6
		# elif 200 <= epoch < 300:
		# 	pretrain_lr = 0.8
		# elif 300 <= epoch < 400:
		# 	pretrain_lr = 1
		# elif 400 <= epoch < 500:
		# 	pretrain_lr = 2
		# else:
		# 	pretrain_lr = 3
		for batch_index in xrange(n_train_batches):
			c.append(pretrain_fns[i](index = batch_index, corruption = corruption_levels[i], lr = pretrain_lr))
		print 'Pre-training layer %i, epoch %d, cost ' %(i, epoch)
		print numpy.mean(c)
		c_all[i, epoch] = numpy.mean(c)

end_time = time.clock()

print >> sys.stderr, ('The pretraining code for file ' +
                      __name__ +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

plt.figure()
# plt.plot(c_all[0],color = 'r')
plt.plot(c_all[1],color = 'b')
plt.show()

print '... getting the finetuning functions'
train_fn, validate_model, test_model = sda.build_finetune_functions(datasets = datasets,
																	batch_size = batch_size,
																	learning_rate = finetune_lr
																	)

minibatch_avg_cost = train_fn(3)
validation_losses = validate_model()
# print '... finetunning the model'
# patience = 100 * n_train_batches # look as this many examples regardless
# patience_increase = 2. # wait this much longer when a new best is found
# improvement_threshold = 0.995 # a relative improvement of this much is considered significant
# validation_frequency = min(n_train_batches, patience / 2) # go through this many minibatches before checking
#                                                           # the network on the validation set; in this case we
#                                                           # check every epoch
# best_validation_loss = numpy.inf
# test_score = 0.
# start_time = time.clock()
# done_looping = False
# epoch = 0

# train_cost_all = []
# valid_cost_all = []
# test_cost_all = []

# while (epoch < training_epochs) and (not done_looping):
# 	epoch +=1
# 	for minibatch_index in xrange(n_train_batches):
# 		minibatch_avg_cost = train_fn(minibatch_index)
# 		iter = (epoch - 1) * n_train_batches + minibatch_index
# 		train_cost_all.append(minibatch_avg_cost)

# 		if (iter + 1) % validation_frequency == 0:
# 			validation_losses = validate_model()
# 			this_validation_loss = numpy.mean(validation_losses)
# 			valid_cost_all.append(this_validation_loss)
# 			print 'epoch %i, minibatch %i/%i, validation_error %f %%' \
# 					% (epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100.)

# 			# if we got the best validation score until now
# 			if this_validation_loss < best_validation_loss:
# 				# improve patience if loss improvement is good enough
# 				if this_validation_loss < best_validation_loss * improvement_threshold:
# 					patience = max(patience, iter * patience_increase)
# 				# save best validation score and iteration numberr
# 				best_validation_loss = this_validation_loss
# 				best_iter = iter

# 				# test it on the test set
# 				test_losses = test_model()
# 				test_score = numpy.mean(test_losses)
# 				print 'epoch %i, minibatch %i/%i, test error of best model %f %%' \
# 						%(epoch, minibatch_index + 1, n_train_batches, test_score * 100.)


# 		if patience <= iter:
# 			done_looping = True
# 			break
# end_time = time.clock()
# print 'Optimization completed with best validation score of %f %%' \
# 		'on iteration %i, with test performance %f %%' \
# 		% (best_validation_loss * 100., best_iter + 1, test_score * 100.)

# print >> sys.stderr ,('The training code ran for %.2f m' % ((end_time - start_time) / 60.))

# plt.figure()
# plt.plot(valid_cost_all,color = 'b')
# plt.show()







































