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
import matplotlib.pyplot as plt
import PCA as PCA   
import LogisticRegression as LOG_REG
from scipy import stats

typeLenghts = []
framerate = 16
minDataPoints = 0.95

#
trialLengths = {}
trialLengths['typeA'] = 269
trialLengths['typeB'] = 269


def PlotXference_AVG():
    global EyeData
    global Events
    #Change these into arrays and prelocate size
    inference = []
    noference = []
    for idx in range(0, len(Events)):
 #       inference.append(FindSlices(EyeData[idx], Events[idx], 'Inference', trialTypes))
#        noference.append(FindSlices(EyeData[idx], Events[idx], 'Noference', trialTypes))        
        inference.append(FindSlices(EyeData[idx], Events[idx], 'Inference', 'typeB',1))
        noference.append(FindSlices(EyeData[idx], Events[idx], 'Noference', 'typeA',1))
    
    fig = plt.figure()
    fig.suptitle('Gaze X position')
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for trial in inference:
        ax.plot(trial)
    ax.set_ylim(0,2000)
    
    ax.set_ylabel('X coordinate of gaze position')
    ax.set_xlabel('Inference trials \n x time course in ms')
    for trial in noference:
        ax2.plot(trial)
    ax2.set_ylim(0,2000)
    ax2.set_xlabel('No inference trials \n x time course in ms')
    
    ticks = ax.get_xticks()*16
    ax.set_xticklabels(ticks.astype(int))
    ax2.set_xticklabels(ticks.astype(int))

    inf_cat  = [1 for i in range (1, len(inference)+1)]
    nof_cat = [0 for i in range (1, len(noference)+1)]
    known_cat = np.hstack((np.array(inf_cat), np.array(nof_cat)))
    ferences = np.vstack((inference, noference)) 
    PlotAverage_X(np.array(inference), np.array(noference))
#    
    
    components = PCA.myPCA(ferences, known_cat)
    components = components *1000
    LOG_REG.logReg(known_cat,components)
    #components_tmp = components *1000
    np.savetxt("eda_pcaResults.csv",np.hstack((known_cat.reshape(len(known_cat),1), components)), delimiter=",")    
    
    
    
  #  return (inference, noference)
    
def PlotXference_IND():
    global EyeData

    global Events
    plt.style.use('ggplot')   
    
    fig = plt.figure()
    fig.suptitle('Individual trials X coordinates')
    inf = fig.add_subplot(121)      
    nof = fig.add_subplot(122)

    inference = []
    noference = []
    inf_cross = []
    nof_cross = []
    
    for idx in range(0, len(Events)):       
        inf_slices, _inf_cross = FilterSlices(EyeData[idx], Events[idx], 'Inference', 'typeB',0)
        inference.append(inf_slices)
        inf_cross.append(_inf_cross)
        nof_slices, _nof_cross = FilterSlices(EyeData[idx], Events[idx], 'Noference', 'typeA',0)
        noference.append(nof_slices)
        nof_cross.append(_nof_cross)
        

        
        for trial in range(0, len(inference[idx])):
            inf.plot(inference[idx][trial, :])
                
        for trial in range(0, len(noference[idx])):
            nof.plot(noference[idx][trial, :])    
            
    nof.set_xlabel('No inference trials')
    inf.set_xlabel('Inference trials')
    inf.set_ylabel('Changes in gaze X position')
    inf.axhline(y = 1200, color='black', linestyle = '--')
    nof.axhline(y = 1200, color='black', linestyle = '--')
    
    ticks = inf.get_xticks()*16
    inf.set_xticklabels(ticks.astype(int))    
    nof.set_xticklabels(ticks.astype(int))    
    
    t,p = stats.ttest_rel(inf_cross, nof_cross)
    print(t)
    print(p)
    
    fig2 = plt.figure()    
    box = plt.subplot(111)
    fig2.suptitle('Average number of center crossing per subject', fontweight = 'bold')
    
    bp1 = box.boxplot([inf_cross, nof_cross], patch_artist=True)

    bp1['boxes'][0].set(color='b', linewidth=0, alpha = 0.5)
    bp1['boxes'][1].set(color='m', linewidth=0, alpha = 0.5)   
    
    box.set_xticklabels(['Inference trials', 'no inference trials'])  
    
    box.set_ylabel('Average amount of crossing per subject', fontweight='bold')
    
    box.set_ylim(-0.01, 2.01)
    return (inf_cross, nof_cross)
    
def PlotAverage_X(inference, noference):
    fig = plt.figure()
    fig.suptitle('X gaze position', fontsize = 18)
    main_plot = fig.add_subplot(111)
    
    main_plot.plot(inference.mean(axis = 0), color = 'b', label='Inference trials')
    main_plot.plot(noference.mean(axis = 0), color = 'r', label='No inference trials')
    
    main_plot.fill_between(np.linspace(0,len(inference[0]), len(inference[0])), inference.mean(axis = 0) - inference.std(axis = 0), inference.mean(axis = 0) + inference.std(axis = 0), color = 'b', alpha = 0.2)
    main_plot.fill_between(np.linspace(0,len(noference[0]), len(noference[0])), noference.mean(axis = 0) - noference.std(axis = 0), noference.mean(axis = 0) + noference.std(axis = 0), color = 'r', alpha = 0.2)
    
#    main_plot.axvline(x=20 , color='black', linestyle = '--', label = 'Stimulus visible')
   # main_plot.axvline(x=len(inference[0]) - len(inference[0])/4.0 , color='black', linestyle = '--')
    main_plot.set_ylabel('X position average and SD', fontsize = 14)
    main_plot.set_xlabel('Time window of the possible inference', fontsize = 14)
    
    main_plot.set_xlim(0,270)
    ticks = main_plot.get_xticks()*16
    main_plot.set_xticklabels(ticks.astype(int))
    legend = main_plot.legend(loc='upper left')
    
def FindSlices(eye, _events, _ference, trialType, returnAverage):
    global framerate
    global minDataPoints
#    fig = plt.figure()
#    fig.suptitle('Gaze during'+ _ference)
#    ax = fig.add_subplot(111)
    global typeLenghts
    #events = events[~events['start'+ _ference].isnull()]
    events = _events[_events['type'] == trialType]

    my_slices = np.zeros(shape = (len(events), 269), dtype = int)
    
    i = 0
    for index, row in events.iterrows():
#Sometimes the slice might be empty because the et data was cut out due to noise or something else (et recording not covering last seconds)        
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
    
    if (returnAverage == 1): 
        return my_slices.mean(axis = 0)
    else:
        return my_slices
                
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
    
def FilterSlices(eye, _events, _ference, trialType, returnAverage):
    global framerate
    global minDataPoints
#    fig = plt.figure()
#    fig.suptitle('Gaze during'+ _ference)
#    ax = fig.add_subplot(111)
    global typeLenghts
    #events = events[~events['start'+ _ference].isnull()]
    events = _events[_events['type'] == trialType]

    my_slices = np.zeros(shape = (len(events), 269), dtype = int)
    
    i = 0
    threshold = 1200
    crossings = []
    for index, row in events.iterrows():
#Sometimes the slice might be empty because the et data was cut out due to noise or something else (et recording not covering last seconds)        
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
            if (all(myslice.avg_x > 10) & all(myslice.avg_y > 0) & all(myslice.avg_y < 1080) & all(myslice.avg_x < 1920)):
#                ax.plot(myslice.avg_x, myslice.avg_y)
 #Round the coordinates (rint) and convert from float to int array (astype)
 #Cut out a few last samples, because their number vary +/- 3 and thus a regular array cannot be made
            #my_slices.append(np.rint(np.array((myslice.avg_x, myslice.avg_y))).astype(int))
                my_slices[i, :] = np.rint(np.array(myslice.avg_x)).astype(int)[0:trialLengths[row['type']]]
                cond = (myslice['avg_x'] < threshold) & (myslice['avg_x'].shift(1) >= threshold)
                crossings.append(len(myslice[cond]))
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
    
    
    mean_crossing = np.array(crossings).mean()
    if (returnAverage == 1): 
        return my_slices.mean(axis = 0)
    else:
        return my_slices, mean_crossing
                

