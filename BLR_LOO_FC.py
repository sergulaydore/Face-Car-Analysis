# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:37:49 2015

@author: sergulaydore
"""

"""
This is a python version of Marios' script:
BLR_LOO_FC.m
"""

sub = 3 # subject number
cohlevel = 4 # coherence level

#%%
print 'Reading events'

"""
readevent.m
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


#%%