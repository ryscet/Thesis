# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:41:37 2015

@author: user
IMPORTANT: EyeData comes fomr another script (BasicAnalysis.py) which loads it into the workspace

1) Run AverageLefRight
2) Run Interpolate 

Trial Keys:
A: noference
B: Inference
"""
import numpy as np
import pandas as pd
import math
import timeit
import matplotlib.pyplot as plt
typeLenghts = []
framerate = 16
minDataPoints = 0.95

#
trialLengths = {}
trialLengths['typeA'] = 269
trialLengths['typeB'] = 269


def PlotXference():
    global EyeData
    global Events
    #Change these into arrays and prelocate size
    inference = []
    noference = []
    for idx in range(0, len(Events)):
 #       inference.append(FindSlices(EyeData[idx], Events[idx], 'Inference', trialTypes))
#        noference.append(FindSlices(EyeData[idx], Events[idx], 'Noference', trialTypes))        
        inference.append(FindSlices(EyeData[idx], Events[idx], 'Inference', 'typeB'))
        noference.append(FindSlices(EyeData[idx], Events[idx], 'Noference', 'typeA'))
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for trial in inference:
        ax.plot(trial)
    ax.set_ylim(0,2000)
    for trial in noference:
        ax2.plot(trial)
    ax2.set_ylim(0,2000)
    return (inference, noference)
    
    
def FindSlices(eye, _events, _ference, trialType):
    global framerate
    global minDataPoints
#    fig = plt.figure()
#    fig.suptitle('Gaze during'+ _ference)
#    ax = fig.add_subplot(111)
    global typeLenghts
    #events = events[~events['start'+ _ference].isnull()]
    events = _events[_events['type'] == trialType]

    my_slices = np.zeros(shape = (len(events), 269), dtype = int)
    
    print(my_slices.shape)
    i = 0
    for index, row in events.iterrows():
#Sometimes the slice might be empty because the et data was cut out due to noise or something else (et recording not covering last seconds)        
        print(row)
        start = eye.index.searchsorted(row['start' + _ference])
        end = eye.index.searchsorted(row['end' + _ference])
#Calculate how long the event lasted        
        timespan = row['end' + _ference] - row['start' + _ference]
        typeLen = (row['type'], _ference, timespan)
#Add each type and its length
        typeLenghts.append(typeLen)
#Calculate the approximate number of lines there should be in gaze data if none are missing
#Approxiamte because the time between each frame is 16.0 ms and not 16.6 as should be in a 60 hz recording
        timespan = (timespan.seconds * 1000 + timespan.microseconds/1000) / framerate
#HACK shouldn't be empty
#IMPORTANT check here if the slice is long enough, i.e. did not happen when eye tracker lost the eyes
        if((eye.ix[start:end].empty == False) & (len(eye.ix[start:end])/ timespan > minDataPoints)):
           # print(_ference, '  ', row['type'])            
            myslice = eye.ix[start:end]
            
 #Round the coordinates (rint) and convert from float to int array (astype)
 #Cut out a few last samples, because their number vary +/- 3 and thus a regular array cannot be made
            #my_slices.append(np.rint(np.array((myslice.avg_x, myslice.avg_y))).astype(int))
            my_slices[i, :] = np.rint(np.array(myslice.avg_x)).astype(int)[0:trialLengths[row['type']]]
            i = i +1
        else:
         #   print(len(eye.ix[start:end])/ timespan)           
             continue  
#            ax.plot(my_slices[-1]['left_x'],my_slices[-1]['left_y'] )
#    ax.set_xlabel('Gaze X')
#    ax.set_ylabel('Gaze Y')
#        
#    ax.set_ylim(400, 1600)
#    ax.set_xlim(0, 2000)
#Filter all the types so each is represented only once
    seen = set()
    typeLenghts = [item for item in typeLenghts if item[0] not in seen and not seen.add(item[0])]
    
    return my_slices.mean(axis = 0)
                
#This function also excludes missing data from the dataframe (condition -10000)               
def AverageLeftRight():
    global EyeData
#Take the average of two eyes to get more accurate gaze position
    for eyes in EyeData:
        eyes['avg_x'] = (eyes['left_x'] + eyes['right_x'])/2        
        eyes['avg_y'] = (eyes['left_y'] + eyes['right_y'])/2
#Do not take the average if one of the eyes was not detected. In that case only use the other eye       
        eyes['avg_x'].loc[eyes.right_x < -100] =  eyes['left_x'].loc[eyes.right_x < -100]
        eyes['avg_y'].loc[eyes.right_y < -100] =  eyes['left_y'].loc[eyes.right_y < -100]
        
        eyes['avg_x'].loc[eyes.left_x < -100] =  eyes['right_x'].loc[eyes.left_x < -100]
        eyes['avg_x'].loc[eyes.left_y < -100] =  eyes['right_y'].loc[eyes.left_y < -100]
#Exclude missing data, TODO interpolate
    for i in range (0, len(EyeData)):
#Das ist unt alternative to interpolation, more conservative (Delete Nans): EyeData[i] = EyeData[i][EyeData[i].avg_x > 0]
        EyeData[i].ix[EyeData[i].avg_x < 0] = np.nan
        valid_col = EyeData[i].dropna()
#Check if first value is not nan, if so substitute with first valid value
        if(math.isnan(EyeData[i].avg_x.iloc[0])):
            EyeData[i].iloc[0] = valid_col.iloc[0]
#Same same if last value is Nan
        if(math.isnan(EyeData[i].avg_x.iloc[-1])):
            EyeData[i].iloc[-1] = valid_col.iloc[-1]
            
# Use my own interpolation function because the built in pandas stuff does not work           
def Interpolate():
    global EyeData
    for i in range (0, len(EyeData)):
        x = np.array(EyeData[i].avg_x, dtype = float)
        y = np.array(EyeData[i].avg_y, dtype = float)
        EyeData[i].avg_x = ActuallyInterpolate(x)
        EyeData[i].avg_y = ActuallyInterpolate(y)
 #       EyeData[i].avg_x = x
  #      EyeData[i].avg_y = y
    
def ActuallyInterpolate(y):    
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y
    


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
    
    

