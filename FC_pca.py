# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:48:52 2015

@author: sergulaydore
"""

import numpy as np
from scipy.io import loadmat
from sklearn import decomposition
import matplotlib.pyplot as plt

""" The idea of autoencoders is similar to the PCA
except the linearity of the manifold. I wonder if can
see some clusters in principle components for
two classes. """

cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'paul21apr04' #'david30apr04' #'brook29sep04' #'an02apr04' #'aaron25jun04' #'jeremy15jul04'

Fs=1000;
StartOffset=-200;
duration=1000-StartOffset;

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
Ntrial = np.shape(EEG['EEG_face'])[2] + np.shape(EEG['EEG_car'])[2]
EEG1 = EEG['EEG_face']
EEG2 = EEG['EEG_car']

time_point = 250

x1 = int(round((time_point-tmin)*Fs/1000));
xbin = x1 + np.arange(Nsample)
data1 = np.mean(np.transpose(EEG1[:,xbin,0:Nface+1]),1) # avg over 30 msec
#data1 = np.transpose(EEG1[:,xbin,0:Nface+1]).reshape(Nface, chan) 
data2 = np.mean(np.transpose(EEG2[:,xbin,0:Ncar+1]),1)
#data2 = np.transpose(EEG2[:,xbin,0:Ncar+1]).reshape(Ncar, chan) 
X_eeg = np.vstack((data1,data2))
y_eeg = np.append(np.ones(Nface), np.zeros(Ncar))
y_ds = y_eeg[range(0,Nface+Ncar, Nsample)]

print 'PCA' + '-'*5

pca = decomposition.PCA(n_components=2, whiten = True)
pca.fit(X_eeg)
X_pca = pca.transform(X_eeg)

#plot data
import seaborn as sns
sns.axes_style("darkgrid")
cols = {0: 'r',1: 'b'}
for idx in range(Ntrial):
    x_1 = X_pca[idx][0]
    x_2 = X_pca[idx][1]
    plt.plot(x_1, x_2,cols[y_ds[idx]]+'o', markersize=13)
    plt.title(('At time %s, blue = Face, red = Car' %time_point), fontsize = 20)
    plt.xlabel('Principle Component 1', fontsize=20)
    plt.ylabel('Principle Component 2', fontsize=20)
#    plt.xlim(-1.1, 1.1)
#    plt.ylim(-1.1, 1.1)




















