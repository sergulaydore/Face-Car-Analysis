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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import shuffle
from data_prep import *
from sklearn import preprocessing
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
    def __init__(self, numpy_rng, theano_rng = None, n_ins = 110, hidden_layers_sizes = [30, 5], 
                n_outs = 2, corruption_levels = [0.1, 0.2]):

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
        self.y_pred = self.logLayer.y_pred

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
        (test_set_x, test_set_y) = datasets[1]

        # compute number of minibatches for training, validatio and testing

        n_test_batches = test_set_x.get_value(borrow = True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index') # index to a minibatch

        # compute the gradients with respect to model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        # updates = [ (param, param - gparam * learning_rate) 
        #            for param, gparam in zip(self.params, gparams)
        #            ]

        updates = gradient_updates_momentum(self.finetune_cost, self.params, learning_rate) 



        train_fn = theano.function( inputs = [index], outputs = self.finetune_cost, updates = updates,
                                    givens = { self.x: train_set_x[ index * batch_size: (index + 1) * batch_size],
                                                self.y: train_set_y[ index * batch_size: (index + 1) * batch_size ]}, 
                                    name = 'train')

        test_score = theano.function( inputs = [], outputs = self.errors,
                                        givens={self.x: test_set_x, self.y: test_set_y}, name = 'test' )

        predict = theano.function (inputs = [], outputs = self.y_pred,
                                    givens = {self.x : test_set_x})

        hidden_output1 = self.sigmoid_layers[0].output
        compute_hidden_output1 = theano.function(inputs = [], outputs = [hidden_output1], 
                                                    givens = {self.x : train_set_x})
#        output1 = compute_hidden_output(X_eeg)


        return train_fn, test_score, predict, compute_hidden_output1


# true_values = test_set_y.owner.inputs[0].get_value() 
def test_sda(train_set_x, test_set_x, train_set_y, test_set_y ):
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    datasets = [(train_set_x, train_set_y),
                (test_set_x, test_set_y)]

    n_train_batches = train_set_x.get_value(borrow = True).shape[0]
    n_train_batches /= batch_size

    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'

    sda = SdA( numpy_rng = numpy_rng)
    print '... getting the pretraining functions'
    pretrain_fns = sda.pretraining_functions(train_set_x = train_set_x, batch_size = batch_size)
    start_time = time.clock()
    corruption_levels = [.1, 0.2, 0.3]
    c_all = np.zeros((sda.n_layers,pretraining_epochs ))
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrain_fns[i](index = batch_index, corruption = corruption_levels[i], lr = pretrain_lr))
            # print 'Pre-training layer %i, epoch %d, cost ' %(i, epoch)
            # print numpy.mean(c)
            c_all[i, epoch] = numpy.mean(c)

    end_time = time.clock()

    # print >> sys.stderr, ('The pretraining code for file ' +
    #                       __name__ +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # plt.figure()
    # # plt.plot(c_all[0],color = 'r')
    # plt.plot(c_all[1],color = 'b')
    # plt.show()

    print '... getting the finetuning functions'
    train_fn, test_model, predict, compute_hidden_output1 = sda.build_finetune_functions(datasets = datasets, batch_size = batch_size, 
                                                        learning_rate = finetune_lr)

    # minibatch_avg_cost = train_fn(3)
    # validation_losses = validate_model()
    # print '... finetunning the model'

    train_cost_all = []
    test_cost_all = []
    acc = []
    
    for epoch in range(training_epochs):

        train_cost_minibatch = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            train_cost_minibatch.append(minibatch_avg_cost)
        train_cost_all.append(numpy.mean(train_cost_minibatch))
        test_losses = test_model()
        test_cost_all.append(test_losses)
    
    prediction = predict()
    hidden_output1 = compute_hidden_output1()
    return prediction, sda, hidden_output1

X_eeg, y_eeg = shuffle(X_eeg, y_eeg, random_state=0)
# pca = decomposition.PCA(n_components=None, copy=True, whiten=True)
# pca.fit(X_eeg)
# X_eeg = pca.transform(X_eeg)
X_eeg = preprocessing.normalize(X_eeg, axis = 1)
finetune_lr=0.1; pretraining_epochs=1000; pretrain_lr=0.08; training_epochs=200; batch_size=10

loo = cross_validation.LeaveOneOut(np.shape(X_eeg)[0])
predictions = []
face_hidden_output1 = None
car_hidden_output1 = None
for train_index, test_index in loo:
    print("TRAIN:", train_index, "TEST:", test_index)
    train_set_x, test_set_x = X_eeg[train_index], X_eeg[test_index]
    train_set_y, test_set_y = y_eeg[train_index], y_eeg[test_index]
    prediction, sda, hidden_output1 = test_sda(train_set_x, test_set_x, train_set_y, test_set_y )
    hidden_output1 = hidden_output1[0]
    if face_hidden_output1 is not None:
        face_hidden_output1 = np.vstack((face_hidden_output1, hidden_output1[train_set_y == 0,:]))
        car_hidden_output1 = np.vstack((car_hidden_output1, hidden_output1[train_set_y == 1,:]))
    else:
        face_hidden_output1 = hidden_output1[train_set_y == 0,:]
        car_hidden_output1 = hidden_output1[train_set_y == 1,:]
    predictions.append(prediction)
    print np.shape(face_hidden_output1)

# train_set_x, test_set_x, train_set_y, test_set_y = cross_validation.train_test_split( X_eeg, y_eeg,
#                                                                                  test_size=0.4, random_state=2)

idx = 0
f = face_hidden_output1[:,idx]
c = car_hidden_output1[:,idx]

plt.figure()
max_data = np.r_[f, c].max()
#bins = np.linspace(0, max_data, max_data + 1)
plt.hist(f, normed=True, color="#6495ED", alpha=.5)
plt.hist(c, normed=True, color="#F08080", alpha=.5)
plt.show()

channel_labels = ['L', 'L', 'L','M', 'M','M','M','M','R','R','R']
fig, axes = plt.subplots(nrows=6, ncols=5, sharex=True, sharey=True)
for idx, ax in enumerate(axes.flat):
    f = face_hidden_output1[:,idx]
    c = car_hidden_output1[:,idx]
    im1 = ax.hist(f, normed=True, color="#6495ED", alpha=.5)
    im1 = ax.hist(c, normed=True, color="#F08080", alpha=.5)
    ax.set_title('Feature ' + str(idx))
    ax.set_adjustable('box-forced')
    # ax.set_xticklabels([0,50, 150, 250, 350, 450,500], rotation='vertical')
    # y = np.arange(0, 11)
    # ax.set_yticks(y)
    # ax.set_yticklabels(channel_labels )

# cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(im, cax=cax, **kw)

fig.text(0.5, 0.04, 'Time', ha='center', va='center')
fig.text(0.04, 0.5, 'Channels', ha='center', va='center', rotation='vertical')
plt.show()

#prediction = test_sda(train_set_x, test_set_x, train_set_y, test_set_y )
print accuracy_score(y_eeg, predictions)

W1 = sda.sigmoid_layers[0].W.get_value(borrow=True) # 110 x 30
W2 = sda.sigmoid_layers[1].W.get_value(borrow=True) # 30 x 5
min_val, max_val = np.min(abs(W1)), np.max(abs(W1))

import matplotlib as mpl
channel_labels = ['L', 'L', 'L','M', 'M','M','M','M','R','R','R']
fig, axes = plt.subplots(nrows=6, ncols=5, sharex=True, sharey=True)
for idx, ax in enumerate(axes.flat):
    my_image = np.transpose(W1[:,idx].reshape(len(timebin_onset), len(channels)))
    im = ax.imshow(abs(my_image), vmin = min_val, vmax = max_val)
    ax.set_title('Feature ' + str(idx))
    ax.set_adjustable('box-forced')
    ax.set_xticklabels([0,50, 150, 250, 350, 450,500], rotation='vertical')
    y = np.arange(0, 11)
    ax.set_yticks(y)
    ax.set_yticklabels(channel_labels )

cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im, cax=cax, **kw)

fig.text(0.5, 0.04, 'Time', ha='center', va='center')
fig.text(0.04, 0.5, 'Channels', ha='center', va='center', rotation='vertical')
plt.show()

fig, ax = plt.plot()
idx = 1
f = face_hidden_output1[:,idx]
c = car_hidden_output1[:,idx]
im1 = ax.hist(f, normed=True, color="#6495ED", alpha=.5)
im1 = ax.hist(c, normed=True, color="#F08080", alpha=.5)
ax.set_title('Feature ' + str(idx))
plt.show()

# test reshape
# plt.figure()
# data=np.arange(110).reshape((11,10))       
# plt.imshow(data)
# plt.title('Main title')
# plt.colorbar()
# plt.show() 


# from utils import tile_raster_images

# try:
#     import PIL.Image as Image
# except ImportError:
#     import Image

# image = Image.fromarray(tile_raster_images(
#         X=sda.dA_layers[0].W.get_value(borrow=True).T,
#         img_shape=(11, 10), tile_shape=(6, 5),
#         tile_spacing=(1, 1)))
# image.save('filters_corruption_first_layer.png')

# image = Image.fromarray(tile_raster_images(
#         X=sda.dA_layers[1].W.get_value(borrow=True).T,
#         img_shape=(30, 1), tile_shape=(1, 5),
#         tile_spacing=(1, 1)))
# image.save('filters_corruption_second_layer.png')

# plt.figure()
# subjects = ['aaron25jun04', 'an02apr04', 'brook29sep04', 'david30apr04', 
#             'jeremy15jul04', 'jeremy29apr04','paul21apr04', 'steve29jun04', 'vivek23jun04']

# subjects = map(lambda x: x[-7:], subjects)

# dl_accuracy = [0.40, 0.797, 0.64, 0.55, 0.764, 0.831, 0.825, 0.66, 0.615]
# pca_10 = [0.466, 0.721, 0.683, 0.572, 0.831, 0.797, 0.6125, 0.651, 0.703]
# pca_30 = [0.531, 0.759, 0.684, 0.607, 0.752, 0.819, 0.65, 0.684, 0.779]
# dl_all_subjects = np.ones(np.shape(dl_accuracy)) * 0.68
# dl_7_subjects = np.ones(np.shape(dl_accuracy)) * 0.69


# plt.plot(range(len(dl_accuracy)), dl_accuracy, 'ro-', label = 'Stacked DAE')
# plt.plot(range(len(dl_accuracy)), dl_all_subjects, 'm0-', label = 'Stacked DAE - all subjects')
# plt.plot(range(len(dl_accuracy)), pca_10, 'bo-', label = 'PCA - 10')
# plt.plot(range(len(dl_accuracy)), pca_30, 'go-', label = 'PCA - 30')
# plt.xticks(range(len(dl_accuracy)), subjects, rotation='vertical', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.title('Performance of Unsupervised + Supervised Algorithms (LOO)')
# plt.legend(loc = 'best')
# plt.subplots_adjust(bottom=0.15)
# plt.show()








































