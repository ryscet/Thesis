# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:39:59 2015

@author: user
Needs to be run after the basic analysis which loads all the data into workspace
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def AverageLeftRight(EyeData):
#Take the average of two eyes to get more accurate gaze position

    for eyes in EyeData:
        eyes['avg_x'] = (eyes['left_x'] + eyes['right_x'])/2        
        eyes['avg_y'] = (eyes['left_y'] + eyes['right_y'])/2
#Do not take the average if one of the eyes was not detected. In that case only use the other eye       
            
        eyes['avg_x'].loc[eyes.right_x < -100] =  eyes['left_x'].loc[eyes.right_x < -100]
        eyes['avg_y'].loc[eyes.right_y < -100] =  eyes['left_y'].loc[eyes.right_y < -100]
        
        eyes['avg_x'].loc[eyes.left_x < -100] =  eyes['right_x'].loc[eyes.left_x < -100]
        eyes['avg_x'].loc[eyes.left_y < -100] =  eyes['right_y'].loc[eyes.left_y < -100]
        
        eyes = eyes.loc[eyes.avg_x > 0]


def PlotXY(EyeData):
    
    for eyes in EyeData:
        fig = plt.figure()
        fig.suptitle('Separate Components')
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.plot(eyes.avg_x)
        ax2.plot(eyes.avg_y)
