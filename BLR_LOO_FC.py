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

sub = 1 # subject number
cohlevel = 30 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'jeremy15jul04'

gaincorrect=1;
Fs=1000;
StartOffset=-200;
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
timebin_onset.extend(range(150,460,10))
timebin_onset.extend(range(500,750,50))
L_timebin = 30 # length of the timebin (ms)
Nsample = int(round(L_timebin/float(1000)*Fs))
Nface = np.shape(EEG['EEG_face'])[2]*Nsample
Ncar = np.shape(EEG['EEG_car'])[2]*Nsample

# LOO loop
tt = 0
print ('LR using time bin %s - %s ms ...')%(timebin_onset[tt],timebin_onset[tt]+L_timebin)
x1 = int(round((timebin_onset[tt]-tmin)*Fs/1000));
xbin = x1 + np.arange(Nsample)
data1 = EEG['EEG_face'][:,xbin,:].reshape(chan, Nface) 
data1 = np.transpose(data1) # 900 x 60
data2 = EEG['EEG_car'][:,xbin,:].reshape(chan, Ncar) 
data2 = np.transpose(data2) # 1200 x 60
X = np.vstack((data1,data2))
y = np.append(np.ones(Nface), np.zeros(Ncar))

# leave one out
Ntrial = (Nface+Ncar)/Nsample
beta, y_LOO ,LOO = [], [], [] # we probably don't need this anymore
                              # thanks to scikit library
acc, predictions, confidence_scores = [], [], []
for k in range(Ntrial):
    print k
    LOO_index = range(k*Nsample, (k+1)*Nsample)
    train_index = list(set(range(Nface+Ncar))-set(LOO_index)) # remove one trial from dataset
                                                              # sets are cool to perform set like operations
                                                              # Here we find A-B for sets A and B
    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression(penalty='l2', C=100000000000000) # cool things start here
                                 # scikit is a popular machine leraning library in python
                                 # good to know this but we will later use theano
                                 # for more complex algorithms
    model = model.fit(X[train_index,:], y[train_index])
    confidence_scores.append( model.decision_function(np.mean(X[LOO_index,:],0)) )
                             # The confidence score for a sample is the signed
                             # distance of that sample to the hyperplane.  
    single_pred = model.predict(np.mean(X[LOO_index,:],0)) 
    print single_pred
    del model
    predictions.append( single_pred ) # compute predictions
                                                                   # we use average over 30 samples
 
acc.append( accuracy_score(y[range(0,np.shape(X)[0], Nsample)],predictions) ) # scikit's accuracy_score function
                                                           # computes Accuracy classification score
                                                           # note that I had to downsample y
                                                           # because I don't need all 2100 samples
                                                           # I only need 70 trials

fpr, tpr, thresholds = roc_curve(y[range(0,2100, Nsample)], confidence_scores)
auc = auc(fpr,tpr)


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()
