
from __future__ import division
from data_prep import *
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
import scipy as sp
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import math
from sklearn.metrics import roc_auc_score, accuracy_score

def normalize(data):
    """
    The normalization happends by
    calculatin (1) mean PER IMAGE (2) std of all images 
    data: n_patches x n_feats
    """
    ## 1. remove patch mean - mean is specific to each image
    ## which depends on the exposure
    data = data - data.mean(axis = 1)[:, sp.newaxis]
    ## 2. truncate to +/- 3 std and scale to -1 to 1
    ## based on the assumption all natural images should have
    ## similiar stds
    data_std = 3. * sp.std(data)
    data = sp.maximum(sp.minimum(data, data_std), -data_std) / data_std
    ## 3. rescale from [-1, 1] to [0.1, 0.9]
    data = (data + 1) * 0.4 + 0.1
    return data

def gradient_updates_momentum(cost, params, learning_rate=0.001, momentum=0.9):

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
         updates = [(param, param-learning_rate * gparam)
                     for param, gparam in zip(self.params, gparams)
                     ]
#         updates = gradient_updates_momentum(cost, self.params)            
                     
         return (cost, updates)  

print 'nTrials x nFeatures ', np.shape(X_eeg)
print 'Target vector ' , np.shape(y_eeg)
print 'Total number of subjects: ', subject_count

""" Generate symbolic variables for input (X and y
represent a minibatch)
"""
X = T.matrix('X') # 2100 x 60 data
y = T.vector('y') # labels, presented as 1D vector of [int] labels

""" Construct the logistic regression class """
rng = np.random.RandomState(1234)
n_hidden = 50
n_visible = np.shape(X_eeg)[1]
da = dA(numpy_rng=rng, input=X,
        n_visible=n_visible, n_hidden=n_hidden)
                   
cost, updates = da.get_cost_updates(corruption_level=0.2,
                            learning_rate=0.01)
                     
                     
train = theano.function(inputs = [X], outputs=cost, updates=updates, allow_input_downcast=True)                     
predict = theano.function(inputs = [X], outputs = da.z)   

""" Leave One Out """
acc = []
batch_size = 100
training_steps = 1000
n_batch = int(math.ceil((Nface + Ncar )/batch_size))
X_eeg = normalize(X_eeg)  
loo = cross_validation.LeaveOneOut(np.shape(X_eeg)[0])

predictions = []
cost_all = []
error_test_all=[]
for train_index, test_index in loo:
	print test_index
	random.shuffle(train_index)
	Xtrain = X_eeg[train_index,:]
#	ytrain = y_eeg[train_index]
	Xtest = X_eeg[test_index,:]
#	ytest = y_eeg[test_index]

	da.W.set_value(np.asarray(
                 rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                                   high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                                   size = (n_visible, n_hidden)
                                   ),
                                    dtype = theano.config.floatX
                            )
              )
              
	da.b_prime.set_value(np.zeros(
	                              n_visible, dtype = theano.config.floatX 
	                             )
	                 )     
	                 
	da.b.set_value(np.zeros(
	                              n_hidden, dtype = theano.config.floatX 
	                             )
	                 )        

	cost_iter = []
	error_test_iter = []
	for idx in range(training_steps):
		cost_batch = []
		for idx_batch in range(n_batch):
			batch_index = range(idx_batch*batch_size,
			                    (idx_batch+1)*batch_size)
			batch_index = filter((lambda x: x<np.shape(X_eeg)[0]-1), batch_index)
			cost = train(Xtrain[batch_index,:])
			cost_batch.append(cost)
		
		test_predict = predict(Xtest)
		error_test_iter.append(np.mean((Xtest-test_predict)**2))
		cost_iter.append(np.mean(cost_batch))

	single_pred = predict(Xtest)
	predictions.append(single_pred)    
	cost_all.append(cost_iter)
	error_test_all.append(error_test_iter)
	# try:
	# 	plt.plot(cost_all[1]); plt.show()
	# except:
	# 	pass

plt.figure
plt.plot(cost_all[1],color = 'r')
plt.plot(error_test_all[1],color = 'b')
plt.show()

plt.figure() 
plt.plot(single_pred[0], color='r')     
plt.xlabel('channels',fontsize=14)
plt.plot(Xtest[0])   

plt.figure()   
plt.scatter(single_pred[0], Xtest[0])  
     































