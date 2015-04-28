# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:45:34 2015

@author: sergulaydore
"""
from scipy.io import loadmat
import numpy as np

gaincorrect=1;
Fs=1000;
StartOffset= -200;
duration=1000-StartOffset;
Unit=1e7;
channels_left = [24, 53, 54] # P3, PO7, PO5
channels_right = [52, 58, 59] # P6, PO6, PO8
channels_middle = [10, 42, 15, 43, 20] # FCz, C1, Cz, C2, CPz
channels = channels_left + channels_middle + channels_right

tmin = StartOffset
timebin_onset = range(0, 500, 50)
#timebin_onset.extend(range(150,460,50))
#timebin_onset.extend(range(500,750,50))
L_timebin = 30 # length of the timebin (ms)
Nsample = int(round(L_timebin/float(1000)*Fs))
x1 = (((np.array(timebin_onset)-tmin)*Fs/1000));
xbinStart = x1 - L_timebin/2
xbinEnd = x1 + L_timebin/2

cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')

import subprocess
p = subprocess.Popen(["ls", path], stdout=subprocess.PIPE)
output, err = p.communicate()
files_in_dir = output.split('\n')

data1 =  None
Nface = 0
Ncar = 0
subject_count = 0
for files in files_in_dir:
	if files.endswith('04'):
		subject_count += 1
		my_subject = files
		print 'Loading pre-processed data for subject %s ' %my_subject
		path_preprocessed = '../preprocessed_for_python/' + my_subject + '/' + 'Coh' + str(cohlevel) + '.mat'

		EEG = loadmat(path_preprocessed)
       # keys: 'eeg_face', 'eeg_car'
       # np.shape(EEG['EEG_face']) = 60 x 1200 x 30
       # EEG['EEG_car'].size = 60 x 1200 x 40

		Nface += np.shape(EEG['EEG_face'])[2]
		Ncar += np.shape(EEG['EEG_car'])[2]
		EEG1 = EEG['EEG_face'][channels,:,:]
		EEG2 = EEG['EEG_car'][channels,:,:]

		EEG1summary =  np.mean(EEG1[:,x1[0]- L_timebin/2:x1[0]+L_timebin/2 +1,0:Nface+1], 1)
		EEG2summary = np.mean(EEG2[:,x1[0]- L_timebin/2:x1[0]+L_timebin/2 +1,0:Ncar+1], 1)
		for xbin in x1[1:]:
			EEG1summary = np.vstack((EEG1summary, np.mean(EEG1[:,xbin- L_timebin/2:xbin+L_timebin/2 +1,0:Nface+1], 1)))
			EEG2summary = np.vstack((EEG2summary, np.mean(EEG2[:,xbin- L_timebin/2:xbin+L_timebin/2 +1,0:Ncar+1], 1)))

		if data1 == None:
			data1 = np.transpose(EEG1summary)
			data2 = np.transpose(EEG2summary)
		else: 
			data1 = np.vstack((data1, np.transpose(EEG1summary)))  # nTrial x nFeature for Face
			data2 = np.vstack((data2, np.transpose(EEG2summary))) # nTrial x nFeature for Car

X_eeg = np.vstack((data1,data2)) #nTrial x nFeature for Face + Car
print 'nTrials x nFeatures ', np.shape(X_eeg)
y_eeg = np.append(np.ones(Nface, dtype=int), np.zeros(Ncar))
print 'Target vector ' , np.shape(y_eeg)
print 'Total number of subjects: ', subject_count
