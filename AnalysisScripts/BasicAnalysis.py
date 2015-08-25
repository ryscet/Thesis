# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:45:46 2015

@author: ryszardcetnarski
"""

import OpenLogs as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
from datetime import datetime


try:
    EyeData
except NameError:
    print ("Results not loaded. Loading results")
    allResults = op.combineSubjects()
    Events = allResults[0]
    EyeData = allResults[1]
    correct_subjects = [i for i in range (1, len(Events) + 1)]

def plotEye(eye):
    
    eye = eye[eye.left_x != -10000.0]
    fig = plt.figure()
    fig.suptitle("Gaze", fontweight='bold')
    ax = fig.add_subplot(111)
    ax.plot(eye.left_x, eye.left_y)
    
    

def accuracyPlot():
    allMeans = np.zeros((len(correct_subjects),2))
    for i in range(len(correct_subjects)):
        sort_events = Events[i].loc[Events[i]['int_category'] == 1]
        cup_events = Events[i].loc[Events[i]['int_category'] == 0]   
        allMeans[i,0]= sort_events['accuracy'].mean()
        allMeans[i,1]= cup_events['sortAnswer'].mean()
        
    plt.style.use('ggplot')    
    fig = plt.figure()    
    box = plt.subplot(111)
    fig.suptitle('Response Accuracy', fontweight = 'bold')
    
    bp1 = box.boxplot([allMeans[:,0],allMeans[:,1]], patch_artist=True)

    bp1['boxes'][0].set(color='b', linewidth=0, alpha = 0.5)
    bp1['boxes'][1].set(color='m', linewidth=0, alpha = 0.5)   
    
    box.set_xticklabels(['good or bad question', 'what is in the box?'])  
    
    box.set_ylabel('Response accuracy (0 = none, 1 = all accurate)', fontweight='bold')
    
    box.set_ylim(0.4, 1.05)
    
    
def CompareHmaps():
    plt.close('all')
    nbins = 100
    inf_allSubjects = np.zeros((nbins, nbins))
    no_allSubjects = np.zeros((nbins, nbins))

    for i in range(len(correct_subjects)):
        currInfHmap = heatMap(i, 'Inference', 'typeB')
        currNoHmap = heatMap(i, 'Noference', 'typeA')
        inf_allSubjects  = inf_allSubjects + currInfHmap 
        no_allSubjects = no_allSubjects + currNoHmap
        PlotHmap(currInfHmap ,currNoHmap, 'Subject hmap: ' + str(i))    
        
    inf_mean = inf_allSubjects/len(correct_subjects)
    no_mean = no_allSubjects/len(correct_subjects)
    
    PlotHmap(inf_mean, no_mean, 'All subjects comparison')    
    
    return inf_allSubjects, no_allSubjects
    
    
def PlotHmap(inf_mean, no_mean, _subtitle):
    plt.style.use('ggplot')
    
    fig = plt.figure()
    fig.suptitle(_subtitle)
    inf = fig.add_subplot(121)    
    no = fig.add_subplot(122)   
    inf.imshow(inf_mean.T)
    no.imshow(no_mean.T)
    
    inf.set_xlabel('inference')
    inf.set_ylabel('screen coordinates')
    no.set_xlabel('no inference')
    
def heatMap(idx, _ference, trialType):
    nbins = 100
    plt.style.use('ggplot')
    #trialTypes = ['typeA', 'typeB']

    slices = FindSlices(EyeData[idx], Events[idx], _ference, trialType)
    slices = pd.concat(slices)
  
    x = np.array(slices['avg_x'], dtype = float)
    y = np.array(slices['avg_y'], dtype = float)
    
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=nbins, range = [[0, 1920], [0, 1080]], normed=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#    fig = plt.figure()
#    fig.suptitle('Gaze heat map during ' + _ference)
#    ax = fig.add_subplot(111)    
#    
#    ax.imshow(heatmap, extent=extent)
#    ax.set_xlabel('x screen position')
#    ax.set_ylabel('y screen position')
#    plt.show()
    return heatmap

def FindSlices(eye, _events, _ference, trialType):
    
    fig = plt.figure()
    fig.suptitle('Gaze during'+ _ference)
    ax = fig.add_subplot(111)    
    
    #events = events[~events['start'+ _ference].isnull()]
    events = _events[_events['type'] == trialType]

    my_slices = []
    im = plt.imread('/Users/user/Desktop/Thesis/exp_screenshot.png')

    for index, row in events.iterrows():
#Sometimes the slice might be empty because the et data was cut out due to noise or something else (et recording not covering last seconds)        
        start = eye.index.searchsorted(row['start' + _ference])
        end = eye.index.searchsorted(row['end' + _ference])
#HACK shouldn't be empty
#IMPORTANT check here if the slice is long enough, i.e. did not happen when eye tracker lost the eyes
        if(eye.ix[start:end].empty == False):       
            ax.imshow(np.flipud(im), origin='lower')
            my_slices.append(eye.ix[start:end])
            ax.plot(my_slices[-1]['avg_x'],1080 - my_slices[-1]['avg_y'], alpha = 0.7)

        else:
         #   print(len(eye.ix[start:end])/ timespan)           
             continue  
    ax.set_xlabel('Gaze X')
    ax.set_ylabel('Gaze Y')
        
    ax.set_ylim(0, 1080)
    ax.set_xlim(0, 1920)
    
    return my_slices


##plt.scatter(x,y,zorder=1)
#plt.imshow(img, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
#plt.show()

            
    
    