# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:04:29 2015

@author: sergulaydore
"""
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import theano
import theano.tensor as T

class LogisticRegression_theano(object):
    """ Multi-class Logistic Regression Class
    
    The logistic regresssion is fully described by a weight matrix: 'W'
    and bias vector 'b'. Classification is done by projecting data points
    onto a set of hyperplanes, the distance to which is used to determine
    a class membership probability.
    """
    
    def __init__(self, input, n_in, rng):
        """ Initialize the parameters of the logistic regression
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                    which the data points lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
#        self.W = theano.shared(
#            value = np.zeros(n_in, dtype=float),
#            name = 'W',
#            borrow = True
#        )
        
        W_values = np.asarray(
            rng.uniform(
                low = -4*np.sqrt(6. / (n_in + 2)),
                high = 4*np.sqrt(6. / (n_in + 2) ),
                size = (n_in,)
            )
        )

        self.W = theano.shared(value = W_values, name = 'W', borrow=True)        
        
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value = 0.,
            name = 'b',
           # borrow = True
        )
        
        """ symbolic expression for computing the matrix of class-membership
        probabilities where:
        W is a matrix where column-k represents the separation hyperplane
        for class-k
        x is a matric where row-j represents input training sample-j
        b is a vector where element-k represent the free parameter of
        hyperplane-k
        """
        
        self.p1 = 1 / (1 + T.exp(-T.dot(input, self.W) - self.b)) # Probability that target = 1
        
        """ symbolic description of how to compute prediction as class
        whose probability is maximal.
        """
        self.y_pred = self.p1>0.5
        self.params = [self.W, self.b]
        
    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        
        type y: theano.tensor.TensorType
        param y: corresponds to a vector that gives for each example
                the correct lable
        """        
        # y.shape[0]: (symbolically) the number of rows in y, i.e., number of
        #examples (call it n) in the minibatch
        
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        #  [0,1,2,..., n-1]
        
        # T.log(self.p_y_given_x) is a matrix of Log=probabilities (call it LP)
        # with one row per example and one column per class.
        
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0, y[0]], LP[1, y[1]], 
        # ..., LP[n-1, y[n-1]]] 
        
        # T.mean(LP[T.arange(y.shape[0]),y]) is the mean (accross minibatch examples)
        # of the elements in v, i.e., the mean log-likelihood across the minibatch
        self.xent = -y * T.log(self.p1) - (1-y) * T.log(1-self.p1)
        return self.xent.mean()
        
    def l2(self):
        return T.sum(self.W ** 2)
        
    def errors(self, y):
        """ Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero one 
        loss over the size of the minibatch
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                the correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
            
        # check if y is of the correct datatype
        #if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s
            # where 1 reprsents a mistake in prediction
        return T.mean(T.neq(self.y_pred, y))
        #else:
        #    raise NotImplementedError()

cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'paul21apr04'

gaincorrect=1;
Fs=1000;
StartOffset=0 #-200;
duration=1000-StartOffset;
Unit=1e7;

print 'Loading pre-processed data' + '-'*10
path_preprocessed = '../preprocessed_for_python/' + my_subject + '/' + 'Coh' + str(cohlevel) + '.mat'
EEG = loadmat(path_preprocessed)
       # keys: 'eeg_face', 'eeg_car'
       # np.shape(EEG['EEG_face']) = 60 x 1200 x 30
       # EEG['EEG_car'].size = 60 x 1200 x 40

print 'Initialization' + '-'*10
chan = np.shape(EEG['EEG_face'])[0]
tmin = StartOffset
timebin_onset = range(0, 150, 50)
timebin_onset.extend(range(150,460,50))
timebin_onset.extend(range(500,750,50))
L_timebin = 30 # length of the timebin (ms)
Nsample = int(round(L_timebin/float(1000)*Fs))
Nface = np.shape(EEG['EEG_face'])[2]*Nsample
Ncar = np.shape(EEG['EEG_car'])[2]*Nsample
EEG1 = EEG['EEG_face']
EEG2 = EEG['EEG_car']
n_features = chan

""" Generate symbolic variables for input (X and y
represent a minibatch)
"""
X = T.matrix('X') # 2100 x 60 data
y = T.vector('y') # labels, presented as 1D vector of [int] labels

""" Construct the logistic regression class """
rng = np.random.RandomState(1234)
classifier = LogisticRegression_theano(input=X, n_in=n_features, rng=rng) 
          
""" The cost we minimize during training is the negative likelihood
of model in symbolic format """
lambda_2 = 1/1e6
cost = classifier.negative_log_likelihood(y) + lambda_2*classifier.l2()

""" BUILDING MODEL"""

""" Learning the model"""
# gradients
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

# specify how to update the parameters of the model 
learning_rate = 0.001
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

# compiling a Theano function 'train_model' that returns the cost, but in
# the same time updates the parameter of the model based on the rules
# defined in updates
train_model = theano.function(
    inputs = [X,y],
    outputs = cost,
    updates = updates
)

""" Testing the Model"""
test_model = theano.function(
    inputs = [X, y],
    outputs = classifier.errors(y)
)

predict = theano.function(
    inputs = [X],
    outputs = classifier.y_pred 
)

""" Leave One Out """
acc = []
# LOO loop
for time_point in timebin_onset:
    
    print ('LR using time bin %s - %s ms ...')%(time_point,time_point+L_timebin)
    x1 = int(round((time_point-tmin)*Fs/1000));
    xbin = x1 + np.arange(Nsample)
    data1 = np.transpose(EEG1[:,xbin,0:Nface+1]).reshape(Nface, chan)  # 900 x 60
    data2 = np.transpose(EEG2[:,xbin,0:Ncar+1]).reshape(Ncar, chan)# 1200 x 60
    X_eeg = np.vstack((data1,data2))
    y_eeg = np.append(np.ones(Nface, dtype=int), np.zeros(Ncar))
    
    Ntrial = (Nface+Ncar)/Nsample
    
    predictions = []
    training_steps = 10000
    cost_all = []
    err_all=[]
    
    for k in range(Ntrial):
        LOO_index = range(k*Nsample, (k+1)*Nsample)
        train_index = list(set(range(Nface+Ncar))-set(LOO_index)) # remove one trial from dataset
    
        Xtrain = X_eeg[train_index,:]
        ytrain = y_eeg[train_index]
        Xtest = np.mean(X_eeg[LOO_index,:],0).reshape(1, chan)
        ytest = y_eeg[LOO_index[0:1]]
        
        classifier.W.set_value(np.asarray(
            rng.uniform(
                low = -4*np.sqrt(6. / (n_features + 2)),
                high = 4*np.sqrt(6. / (n_features + 2) ),
                size = (n_features,)
            )
        ))
        classifier.b.set_value(0.)
        cost_single = []
        for idx in range(training_steps):
            cost = train_model(Xtrain, ytrain)
            cost_single.append(cost)
            
        single_pred = predict(Xtest)
        predictions.append(single_pred)    
        cost_all.append(cost_single)
        err = test_model(Xtest, ytest)
        err_all.append(err)
      
    acc.append( accuracy_score(y_eeg[range(0,np.shape(X_eeg)[0], Nsample)],predictions) ) # scikit's accuracy_score function
                                                           # computes Accuracy classification score
                                                           # note that I had to downsample y
                                                           # because I don't need all 2100 samples
                                                         # I only need 70 trials  
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.axes_style("darkgrid")
plt.plot(timebin_onset, acc, sns.xkcd_rgb["pale red"], lw=3)
plt.ylabel('Accuracies',fontsize=14)
plt.xlabel('time(msec)',fontsize=14)
plt.title('Leave One Out ; Subject ' + my_subject + '; Coherence level ' + str(cohlevel))    