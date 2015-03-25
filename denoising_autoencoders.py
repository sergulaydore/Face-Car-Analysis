# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:32:32 2015

@author: sergulaydore
"""
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
                     
         return (cost, updates)            
                     

cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'paul21apr04'

gaincorrect=1;
Fs=1000;
StartOffset= -200;
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
learning_rate = 0.03     
                
""" Generate symbolic variables for input (X and y
represent a minibatch)
"""
X = T.matrix('X') # 2100 x 60 data
y = T.vector('y') # labels, presented as 1D vector of [int] labels

""" Construct the logistic regression class """
rng = np.random.RandomState(1234)
n_hidden = 20
n_visible = n_features
da = dA(numpy_rng=rng, input=X,
        n_visible=n_visible, n_hidden=n_hidden)
                   
cost, updates = da.get_cost_updates(corruption_level=0.2,
                            learning_rate=learning_rate)
                     
                     
train = theano.function(inputs = [X], outputs=cost, updates=updates, allow_input_downcast=True)                     
predict = theano.function(inputs = [X], outputs = da.z)   

""" Leave One Out """
acc = []
batch_size = 100
n_batch = (Nface + Ncar - L_timebin)/batch_size
# LOO loop
plt.figure()
for time_point in [120]: #timebin_onset: 
    print ('LR using time bin %s - %s ms ...')%(time_point,time_point+L_timebin)
    x1 = int(round((time_point-tmin)*Fs/1000));
    xbin = x1 + np.arange(Nsample)
    data1 = np.transpose(EEG1[:,xbin,0:Nface+1]).reshape(Nface, chan)  # 900 x 60
    data2 = np.transpose(EEG2[:,xbin,0:Ncar+1]).reshape(Ncar, chan)# 1200 x 60
    X_eeg = np.vstack((data1,data2))
    y_eeg = np.append(np.ones(Nface, dtype=int), np.zeros(Ncar))
    
    Ntrial = (Nface+Ncar)/Nsample
    
    predictions = []
    training_steps = 100
    cost_all = []
    err_all=[]
    
    for k in range(Ntrial):
        LOO_index = range(k*Nsample, (k+1)*Nsample)
        train_index = list(set(range(Nface+Ncar))-set(LOO_index)) # remove one trial from dataset
        random.shuffle(train_index)
        Xtrain = X_eeg[train_index,:]
        ytrain = y_eeg[train_index]
        Xtest = np.mean(X_eeg[LOO_index,:],0).reshape(1, chan)
        ytest = y_eeg[LOO_index[0:1]]
                  
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
        for idx in range(training_steps):
            cost_batch = []
            for idx_batch in range(n_batch):
                batch_index = range(idx_batch*batch_size,
                                    (idx_batch+1)*batch_size)
                cost = train(Xtrain[batch_index,:])
                cost_batch.append(cost)
            cost_iter.append(np.mean(cost_batch))
            
        single_pred = predict(Xtest)
        predictions.append(single_pred)    
        cost_all.append(cost_iter)
        try:
            plt.plot(cost_all[1])
        except:
            pass

#    acc.append( accuracy_score(y_eeg[range(0,np.shape(X_eeg)[0], Nsample)],predictions) ) # scikit's accuracy_score function
                                                           # computes Accuracy classification score
                                                           # note that I had to downsample y
                                                           # because I don't need all 2100 samples
plt.figure() 
plt.plot(single_pred[0])     

plt.figure() 
plt.plot(Xtest[0])                                                    # I only need 70 trials  
#import matplotlib as mpl
#
#import seaborn as sns
#
#plt.figure()
#sns.axes_style("darkgrid")
#plt.plot(timebin_onset, acc, sns.xkcd_rgb["pale red"], lw=3)
#plt.ylabel('Accuracies',fontsize=14)
#plt.xlabel('time(msec)',fontsize=14)
#plt.title('Leave One Out ; Subject ' + my_subject + '; Coherence level ' + str(cohlevel))                
#                     
                     
                     
                     
                                                     
