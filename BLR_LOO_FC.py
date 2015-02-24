# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:37:49 2015

@author: sergulaydore

This is a python version of Marios' script: BLR_LOO_FC.m
"""


import numpy as np
from scipy.io import loadmat

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

print 'Loading EEG data set' + '-'*44
filename_filtered = path + my_subject + '/facecoherence_filtered'
channel=range(2,62); # scalp electrodes only
       # BE CAREFUL with this, indices might be different in python
gaincorrect=1;
Fs=1000;
StartOffset=-200;
duration=1000-StartOffset;
Unit=1e7;

print 'Reading header file'+ '-'*20
import pandas as pd
header = pd.read_csv(filename_filtered + '.hdr', delimiter=';')
D = header.loc[0][0] # number of channels in filename
N = header.loc[1][0] # number of samples in data file
fs = 1/(header.loc[2][0]) # sampling rate
gain = header.loc[3][0] # gain

#function [D,N,fs,gain] = readheader(filename)
#
#% [D,N,fs,gain] = readheader(file)
#
#fid = fopen([filename '.hdr'],'r');
#line = fgetl(fid);
#line = fgetl(fid); D = sscanf(line,'%d');
#line = fgetl(fid); N = sscanf(line,'%d');
#line = fgetl(fid); fs = 1/sscanf(line,'%f');
#line = fgetl(fid); gain = sscanf(line,'%f');
#fclose(fid);

#% Load EEG data
#% [eegdata,D,N,fs,gain,events]=readcogdata(filename,channel,gaincorrect,duration,offset,m,show);
#%
#% Input:
#% 'filename'    - [string] full path to data file
#% 'channel'     - channels to read [all channels]
#% 'gaincorrect' - 1 - apply gain; 0 - do no apply gain [1] 
#% 'duration'    - number of samples to read [all samples]
#% 'offset'      - sample number to begin reading [1]
#% 'm'           - output of calibrate.m to discretize event channel
#% 'show'        - 1 - display progress; 0 - do not display progress [1]
#%
#% Output:
#% 'eegdata'     - [channels x samples] EEG data
#% 'D'           - number of channels in filename
#% 'N'           - number of samples in data file 
#% 'fs'          - sampling rate
#% 'gain'        - gain
#% 'events'      - discretized eegdata (useful for event channel)
#%    
#% Example:
#% [m,s]= calibrate(fileevent); % e.g. fileevent=jumptest 
#% [D,N,fs,gain] = readheader(filein);
#% [eventchannel,D,N,fs,gain,discreteeventchannel]=readcogdata(filein,64,0,N,1,m,1);

#eventindices=StimOffsets{1};
#eegdata1=readcogdata(filename,channel,gaincorrect,duration,eventindices+StartOffset);
#EEG1=reshape(eegdata1,length(channel),duration,[])*Unit;
#eventindices=StimOffsets{2};
#eegdata2=readcogdata(filename,channel,gaincorrect,duration,eventindices+StartOffset);
#EEG2=reshape(eegdata2,length(channel),duration,[])*Unit;




















