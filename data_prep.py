# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:45:34 2015

@author: sergulaydore
"""
from scipy.io import loadmat
import numpy as np

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
timebin_onset = range(0, 500, 50)
#timebin_onset.extend(range(150,460,50))
#timebin_onset.extend(range(500,750,50))
L_timebin = 30 # length of the timebin (ms)
Nsample = int(round(L_timebin/float(1000)*Fs))
Nface = np.shape(EEG['EEG_face'])[2]
Ncar = np.shape(EEG['EEG_car'])[2]
EEG1 = EEG['EEG_face']
EEG2 = EEG['EEG_car']
n_features = chan

x1 = (((np.array(timebin_onset)-tmin)*Fs/1000));
xbinStart = x1 - L_timebin/2
xbinEnd = x1 + L_timebin/2
EEG1summary =  np.mean(EEG1[:,x1[0]- L_timebin/2:x1[0]+L_timebin/2 +1,0:Nface+1], 1)
EEG2summary = np.mean(EEG2[:,x1[0]- L_timebin/2:x1[0]+L_timebin/2 +1,0:Nface+1], 1)
for xbin in x1[1:]:
    EEG1summary = np.vstack((EEG1summary, np.mean(EEG1[:,xbin- L_timebin/2:xbin+L_timebin/2 +1,0:Nface+1], 1)))
    EEG2summary = np.vstack((EEG2summary, np.mean(EEG2[:,xbin- L_timebin/2:xbin+L_timebin/2 +1,0:Ncar+1], 1)))

#print np.shape(EEG1summary)
#print np.shape(EEG2summary)

data1 = np.transpose(EEG1summary)  # nTrial x nFeature for Face
data2 = np.transpose(EEG2summary) # nTrial x nFeature for Car

X_eeg = np.vstack((data1,data2)) #nTrial x nFeature for Face + Car
print np.shape(X_eeg)
y_eeg = np.append(np.ones(Nface, dtype=int), np.zeros(Ncar))
print np.shape(y_eeg)
