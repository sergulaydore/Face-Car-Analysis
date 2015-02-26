# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:37:49 2015

@author: sergulaydore

This is a python version of Marios' script: BLR_LOO_FC.m
"""


import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

sub = 1 # subject number
cohlevel = 30 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'jeremy15jul04'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print 'Reading events'+ '-'*50

#def readevents(my_subject, cohlevel, FC, path, subjects):
# of course wee need to read matlab files
# luckily we can use scipy's loadmat function


# this is dictionary with keys
#['EarlyComponent', 'SelectTime','LateComponent', 'subjectAlpha',
# 'subjectIC', 'SelectCoh', '__header__', '__globals__',
# 'badchan', 'subjects', '__version__', 'rejections']

single_subject = loadmat(path + my_subject + '/events_'+ my_subject + '.mat')
sub = np.where(subjects['subjects'].flatten()==[my_subject])

# this is dictionary with keys
# ['face_responses', 'face_trials', 'correctstimevents', 'coh_values',
# 'incorrectstimevents', 'coherence', '__header__',
# 'reaction_times_idxs', '__globals__', 'responsetimes',
# 'car_responses', 'face_idxs', 'car_trials', 'car_idxs', '__version__']

# GLOBAL VARIABLES
rej = subjects['rejections'][0][sub]

# VARIABLES SPECIFIC TO A SUBJECT
coh_values = single_subject['coh_values'][0]
coh_idx = np.where(single_subject['coh_values'][0] == cohlevel)[0][0]
responsetimes = single_subject['responsetimes'][0]

correctstimevents = single_subject['correctstimevents'].reshape(4,12)
incorrectstimevents = single_subject['incorrectstimevents'].reshape(4,12)    
ct_face = np.empty(0)
ct_car = np.empty(0)
ict_face = np.empty(0)
ict_car = np.empty(0)
stim_offset_face = np.empty(0)
stim_offset_car = np.empty(0)
n_trials_face = np.empty(0)
n_trials_car = np.empty(0)
for row_idx in range(len(correctstimevents)): 
    ct_face = np.append(ct_face,correctstimevents[:, coh_idx][row_idx][0])
    ct_car = np.append(ct_car,correctstimevents[:, len(coh_values) + coh_idx][row_idx][0])
    ict_face = np.append(ict_face,incorrectstimevents[:, coh_idx][row_idx][0])
    ict_car = np.append(ict_car,incorrectstimevents[:, len(coh_values) + coh_idx][row_idx][0])   
    stim_offset_face = np.append(stim_offset_face, single_subject['face_idxs'][0][coh_idx][0][row_idx][0])
    stim_offset_car = np.append(stim_offset_face, single_subject['car_idxs'][0][coh_idx][0][row_idx][0])
    n_trials_face = np.append(n_trials_face,
                              len(single_subject['face_idxs'].flatten()[coh_idx].flatten()[row_idx].flatten()))
    n_trials_car = np.append(n_trials_car,
                              len(single_subject['car_idxs'].flatten()[coh_idx].flatten()[row_idx].flatten()))
    
artifact_rej_face = np.ones(len(stim_offset_face))
artifact_rej_face[rej[0][0].flatten()[coh_idx].flatten()]=0
artifact_rej_car = np.ones(len(stim_offset_car))
artifact_rej_car[rej[0][1].flatten()[coh_idx].flatten()]=0
CorrectRespFace = [stim_offset_face[idx] in ct_face for idx in range(len(stim_offset_face)) ]
CorrectRespCar = [stim_offset_car[idx] in ct_car for idx in range(len(stim_offset_car)) ]   
 
events = dict()

events['F'] = {'idxs': single_subject['face_idxs'].flatten(),
               'StimOffsets': stim_offset_face,
               'RTs': responsetimes[coh_idx][0][0][0],
               'allcorrects': ct_face,
               'allincorrects': ict_face,
               'CorrectResp': CorrectRespFace,
               'ArtifactRej':artifact_rej_face,
               'nTrials': n_trials_face
               }
events['C'] = {'idxs': single_subject['car_idxs'].flatten(),
               'StimOffsets': stim_offset_car,
               'RTs': responsetimes[len(coh_values) + coh_idx][0][0][0],
               'allcorrects': ct_car,
               'allincorrects': ict_car,
               'CorrectResp': CorrectRespCar,
               'ArtifactRej': artifact_rej_car,
               'nTrials': n_trials_car
               }
#    return events
#events = readevents(my_subject, cohlevel, FC, path, subjects)    
print '-'*20 + 'Reading events completed' + '-'*20
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load dataset

# Probably don't need this anymore
print 'Loading EEG data set' + '-'*44
filename_filtered = path + my_subject + '/facecoherence_filtered'
channel=range(2,62); # scalp electrodes only
       # BE CAREFUL with this, indices might be different in python
gaincorrect=1;
Fs=1000;
StartOffset=-200;
duration=1000-StartOffset;
Unit=1e7;

# Probably don't need this anymore
print 'Reading header file'+ '-'*20
import pandas as pd
header = pd.read_csv(filename_filtered + '.hdr', delimiter=';')
D = header.loc[0][0] # number of channels in filename
N = header.loc[1][0] # number of samples in data file
fs = 1/(header.loc[2][0]) # sampling rate
gain = header.loc[3][0] # gain

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
tt = 2
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
    LOO_index = range((k-1)*Nsample, k*Nsample)
    train_index = list(set(range(Nface+Ncar))-set(LOO_index)) # remove one trial from dataset
                                                              # sets are cool to perform set like operations
                                                              # Here we find A-B for sets A and B
    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression() # cool things start here
                                 # scikit is a popular machine leraning library in python
                                 # good to know this but we will later use theano
                                 # for more complex algorithms
    model = model.fit(X[train_index,:], y[train_index])
    confidence_scores.append( model.decision_function(np.mean(X[LOO_index,:],0)) )
                             # The confidence score for a sample is the signed
                             # distance of that sample to the hyperplane.  
    predictions.append( model.predict(np.mean(X[LOO_index,:],0)) ) # compute predictions
                                                                   # we use average over 30 samples
 
acc.append( accuracy_score(y[range(0,2100, Nsample)],predictions) ) # scikit's accuracy_score function
                                                           # computes Accuracy classification score
                                                           # note that I had to downsample y
                                                           # because I don't need all 2100 samples
                                                           # I only need 70 trials

fpr, tpr, thresholds = roc_curve(y[range(0,2100, Nsample)], confidence_scores)
auc = auc(fpr,tpr)


















