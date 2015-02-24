# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:37:49 2015

@author: sergulaydore

This is a python version of Marios' script: BLR_LOO_FC.m
"""

import numpy as np

sub = 1 # subject number
cohlevel = 30 # coherence level
FC = 'FC' # face/car trials
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print 'Reading events'

"""
This is a Python version of readevent.m
read stimulus offsets from events file
Input:
     sub: subject #
     FC: ['F'|'C'|'FC'] face/car trials
     cohlevel: [20|25|30|35|40|45] coherence levels
Output:
     StimOffsets: cell array, stimulus offsets of each cohlevel
     RTs: cell array, reaction time of each cohlevel
     CorrectResp: cell array, each cell element is a logical array, 
                  indicating correct responses
     ArtifactRej:
     nTrials: number of trials per block at each cohlevel,
              4*(1 or 2)*(n cohleves)
"""

# of course wee need to read matlab files
# luckily we can use scipy's loadmat function
path = '../FC_Mario/rawdata/' # aaron25jun04/events_aaron25jun04.mat'
from scipy.io import loadmat
subjects = loadmat(path + 'subjects.mat')
# this is dictionary with keys
#['EarlyComponent', 'SelectTime','LateComponent', 'subjectAlpha',
# 'subjectIC', 'SelectCoh', '__header__', '__globals__',
# 'badchan', 'subjects', '__version__', 'rejections']

single_subject = loadmat(path + 'jeremy15jul04/events_jeremy15jul04.mat') 
# this is dictionary with keys
# ['face_responses', 'face_trials', 'correctstimevents', 'coh_values',
# 'incorrectstimevents', 'coherence', '__header__',
# 'reaction_times_idxs', '__globals__', 'responsetimes',
# 'car_responses', 'face_idxs', 'car_trials', 'car_idxs', '__version__']

# GLOBAL VARIABLES
rej = subjects['rejections'][0][sub-1]

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
    ict_face = np.append(ict_face,correctstimevents[:, coh_idx][row_idx][0])
    ict_car = np.append(ict_car,correctstimevents[:, len(coh_values) + coh_idx][row_idx][0])   
    stim_offset_face = np.append(stim_offset_face, single_subject['face_idxs'][0][coh_idx][0][row_idx][0])
    stim_offset_car = np.append(stim_offset_face, single_subject['car_idxs'][0][coh_idx][0][row_idx][0])
    n_trials_face = np.append(n_trials_face,
                              len(single_subject['face_idxs'].flatten()[coh_idx].flatten()[row_idx].flatten()))
    n_trials_car = np.append(n_trials_car,
                              len(single_subject['car_idxs'].flatten()[coh_idx].flatten()[row_idx].flatten()))
    
artifact_rej_face = np.ones(len(stim_offset_face))
artifact_rej_face[rej[0][coh_idx].flatten()]=0
artifact_rej_car = np.ones(len(stim_offset_car))
artifact_rej_car[rej[1][coh_idx].flatten()]=0
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
print '-'*20 + 'Reading events completed' + '-'*20
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%