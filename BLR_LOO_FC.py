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
cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'paul21apr04' #'david30apr04' #'brook29sep04' #'an02apr04' #'aaron25jun04' #'jeremy15jul04'

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
timebin_onset.extend(range(150,500,50))
timebin_onset.extend(range(500,750,50))
L_timebin = 30 # length of the timebin (ms)
Nsample = int(round(L_timebin/float(1000)*Fs))
Nface = np.shape(EEG['EEG_face'])[2]*Nsample
Ncar = np.shape(EEG['EEG_car'])[2]*Nsample


mean_scores = []
acc_all = []
auc_all = []
EEG1 = EEG['EEG_face']
EEG2 = EEG['EEG_car']
acc = []
# LOO loop
for time_point in timebin_onset:
    print ('LR using time bin %s - %s ms ...')%(time_point,time_point+L_timebin)
    x1 = int(round((time_point-tmin)*Fs/1000));
    xbin = x1 + np.arange(Nsample)
    data1 = np.transpose(EEG1[:,xbin,0:Nface+1]).reshape(Nface, chan) 
 #   data1 = np.transpose(data1) # 900 x 60
    data2 = np.transpose(EEG2[:,xbin,0:Ncar+1]).reshape(Ncar, chan) 
#    data2 = np.transpose(data2) # 1200 x 60
    X = np.vstack((data1,data2))
    y = np.append(np.ones(Nface), np.zeros(Ncar))
    downsampled = np.array(range(0,np.shape(X)[0], Nsample))

    # leave one out
    Ntrial = (Nface+Ncar)/Nsample
    
    #evaluate the model using 10-fold cross-validation
#    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
#
#    mean_scores.append(scores.mean())
    
    predictions, confidence_scores = [], []
    for k in range(Ntrial):
        LOO_index = range(k*Nsample, (k+1)*Nsample)
        train_index = list(set(range(Nface+Ncar))-set(LOO_index)) # remove one trial from dataset
                                                                  # sets are cool to perform set like operations
                                                                  # Here we find A-B for sets A and B
        # instantiate a logistic regression model, and fit with X and y
        model = LogisticRegression(penalty='l2',C=1000000) # cool things start here
                                     # scikit is a popular machine leraning library in python
                                     # good to know this but we will later use theano
                                     # for more complex algorithms
        model.fit(X[train_index,:], y[train_index])
#        confidence_scores.append( model.decision_function(np.mean(X[LOO_index,:],0)) )
                                 # The confidence score for a sample is the signed
                                 # distance of that sample to the hyperplane. 
        single_pred = model.predict(np.mean(X[LOO_index,:],0)) 
#        print single_pred
 #       del model
        predictions.append(single_pred) # compute predictions
                                                                   # we use average over 30 samples

    acc.append( accuracy_score(y[range(0,np.shape(X)[0], Nsample)],predictions) ) # scikit's accuracy_score function
                                                           # computes Accuracy classification score
                                                           # note that I had to downsample y
                                                           # because I don't need all 2100 samples
                                                           # I only need 70 trials

#fpr, tpr, thresholds = roc_curve(y[range(0,np.shape(X)[0], Nsample)], confidence_scores)
#auc_all.append(auc(fpr,tpr))

#print model.predict_proba(X[LOO_index,:])
#print model.coef_
#print model.intercept_

from ggplot import *
p_val_features = pd.DataFrame({
      'accuracies': acc,
      'time': timebin_onset
   })
print ggplot(p_val_features, aes(x ='time', y= 'accuracies')) + \
      geom_line() + ggtitle('Leave One Out' ) + \
      labs('time (msec)','Accuracies') + \
      ylim(0,1) 
      

#
#



##from sklearn.cross_validation import KFold
#from sklearn.cross_validation import LeaveOneOut
##kf = KFold(30, 70)
#loo = LeaveOneOut(Ntrial)
#
#downsampled = np.array(range(0,np.shape(X)[0], Nsample))
#predictions = []
#for train, test in loo:
# #   print test
#    model = LogisticRegression(C=10000000000.0, penalty='l2') 
#    model.fit(X[downsampled[train[:]],:], y[downsampled[train[:]]])
#    single_pred = model.predict(X[downsampled[test[:]],:])
#    del model
##    print single_pred
#    predictions.append(single_pred)
#          
















