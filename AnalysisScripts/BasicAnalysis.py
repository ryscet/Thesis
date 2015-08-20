# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:45:46 2015

@author: ryszardcetnarski
"""

import OpenLogs as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



try:
    EyeData
except NameError:
    print ("Results not loaded. Loading results")
    allResults = op.combineSubjects()
    Events = allResults[0]
    EyeData = allResults[1]



def plotEye(eye):
    
    eye = eye[eye.left_x != -10000.0]
    fig = plt.figure()
    fig.suptitle("Gaze", fontweight='bold')
    ax = fig.add_subplot(111)
    ax.plot(eye.left_x, eye.left_y)
    
    
def FindSlices(eye, events, _ference):
        
    eye = eye[eye['left_x'] != -10000 ]
    my_slices = []
    fig = plt.figure()
    fig.suptitle('Gaze during'+ _ference)
    ax = fig.add_subplot(111)

    events = events[~events['start'+ _ference].isnull()]
    for index, row in events.iterrows():
        
        start = eye.index.searchsorted(row['start' + _ference])
        end = eye.index.searchsorted(row['end' + _ference])
                  
        my_slices.append(eye.ix[start:end])

        ax.plot(my_slices[-1]['left_x'],my_slices[-1]['left_y'] )
    ax.set_xlabel('Gaze X')
    ax.set_ylabel('Gaze Y')
        
    ax.set_ylim(400, 1600)
    ax.set_xlim(0, 2000)
    return my_slices
            
def accuracyPlot():
    allMeans = np.zeros((len(correct_subjects),2))
    for i in range(len(correct_subjects)):
        sort_events = Events[i].loc[Events[i]['int_category'] == 1]
        cup_events = Events[i].loc[Events[i]['int_category'] == 0]   
        allMeans[i,0]= sort_events['accuracy'].mean()
        allMeans[i,1]= cup_events['sortAnswer'].mean()
        
    
    fig = plt.figure()
    fig.suptitle('Response accuracy')
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
        
    x= len(correct_subjects)
    width = 0.25
    ax.bar(x, allMeans[0,0], width)
#    ax3.set_xticks(x+width)
#    ax3.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    #print(sort_events['accuracy'].mean())
   # print(cup_events['_sortAnswer'].mean())
    
def heatMap(idx, _ference):
    nbins = 100

    plt.style.use('ggplot')
    slices = FindSlices(EyeData[idx], Events[idx], _ference)
    slices = pd.concat(slices)
  
    x = np.array(slices['left_x'], dtype = float)
    y = np.array(slices['left_y'], dtype = float)
    
    
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
    
def CompareHmaps():
    plt.close('all')
    nbins = 100
    inf_allSubjects = np.zeros((nbins, nbins))
    no_allSubjects = np.zeros((nbins, nbins))
    for i in range(len(correct_subjects)):
        inf_allSubjects  = inf_allSubjects  + heatMap(i, 'Inference')
        no_allSubjects = no_allSubjects + heatMap(i, 'Noference')
        
    inf_mean = inf_allSubjects/len(correct_subjects)
    no_mean = no_allSubjects/len(correct_subjects)
    
    plt.style.use('ggplot')
    
    fig = plt.figure()
    fig.suptitle('Gaze heat map comparison')
    inf = fig.add_subplot(121)    
    no = fig.add_subplot(122)   
    inf.imshow(inf_mean.T)
    no.imshow(no_mean.T)
    
    inf.set_xlabel('inference')
    inf.set_ylabel('screen coordinates')
    no.set_xlabel('no inference')
        
    #return inf_allSubjects, no_allSubjects

            
    
    