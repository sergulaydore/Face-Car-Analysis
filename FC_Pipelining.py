# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:04:08 2015

@author: whayinhsu
"""

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import cross_val_score
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
#Data Preparation
cohlevel = 45 # coherence level
FC = 'FC' # face/car trials
#path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'
path = '../FC_Mario/rawdata/'

subjects = loadmat(path + 'subjects.mat')
my_subject = 'paul21apr04'

gaincorrect=1;
Fs=1000;
StartOffset= -200;
duration=1000-StartOffset;
Unit=1e7;

print 'Loading pre-processed data' + '-'*10
#path_preprocessed = '../preprocessed_for_python/' + my_subject + '/' + 'Coh' + str(cohlevel) + '.mat'
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
Ntrial = Nface + Ncar
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

#PCA
pca = decomposition.PCA(n_components=20,whiten=True)
pca.fit(X_eeg)
X_pca = pca.transform(X_eeg)

#Leave-One-Out Loop
acc = []
predictions, confidence_scores = [], []
for k in range(Ntrial):
    LOO_Index = k
    Train_Index = list(np.setdiff1d(range(Ntrial),k))
    model = LogisticRegression(penalty='l1',C=1.5)
    model.fit(X_pca[Train_Index,:], y_eeg[Train_Index])
    single_pred = model.predict(X_pca[LOO_Index])
    predictions.append(single_pred)

acc.append(accuracy_score(y_eeg, predictions))
print acc

coefficients = model.coef_.flatten()
bestpcacomponent_indices = np.where(abs(coefficients)>0.85)[0]
#
for componentindex in bestpcacomponent_indices:
    pcacomponent = pca.components_[componentindex].reshape(len(timebin_onset), chan)
    plt.figure()
    plt.imshow(np.transpose(abs(pcacomponent)))
    plt.colorbar()
    plt.title('Coefficient %s'%coefficients[componentindex])

#eigenfaces = pca.components_.reshape((20, chan, len(timebin_onset)))
#plt.figure()
#plt.plot(np.transpose(eigenfaces[componentindex]))
#plt.show()

#def plot_gallery(images, titles, h, w,
#                 bestpcacomponent_indices=bestpcacomponent_indices, 
#                 n_row=2, n_col=3):
#    """Helper function to plot a gallery of portraits"""
#    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#    i = 0
#    for c_index in bestpcacomponent_indices:
#        plt.subplot(n_row, n_col, i + 1)
#        plt.imshow(np.transpose(images[c_index].reshape((h, w))))
#        plt.title(titles[i], size=12)
#        i+=1
#        
#eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_gallery(eigenfaces, eigenface_titles, chan, len(timebin_onset))