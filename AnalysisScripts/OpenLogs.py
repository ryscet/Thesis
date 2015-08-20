# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:15:38 2015

@author: ryszardcetnarski
"""
import pandas as pd
import datetime as dt
import numpy as np
import os.path
import time
import os
import subprocess

correct_subjects = [1,2,3,4,5,6,7,8,9,10]
def combineSubjects():
    allEvents = []
    allEye = []
    for i in range(1, 17):
        allEye.append(openEye(i))
        allEvents.append(openLogs(i, allEye[-1].index[0]))
        print(i)
    return allEvents, allEye


def openLogs(subjectID, originalTime):
    pd.options.mode.chained_assignment = None
    _path = "/Users/ryszardcetnarski/Desktop/MasterForSync/Organized_Results/Events/Subject_" + str(subjectID) + "/events.csv"
       
    
    file_time = originalTime
    
    f = open(_path)
    lines = f.readlines()
    lines = lines [2::]
    f.close()
    filterLines = []
    my_index = []
    startIdx = []
    endIdx = []
    for idx, line in enumerate(lines):
#Only look for 72 trials (that is the max amount of trials), the 73rd begun to save when experiment was ending thus eslting in error ('StartingTrial' without 'Question')
        if len(startIdx) < 71: 
            if 'StartingTrial' in line: 
                startIdx.append(idx -1)
        if len(endIdx) < 71:
            if 'Question' in line:
                endIdx.append(idx+2)
                
    print(len(startIdx))
    print(len(endIdx))
    limits = np.vstack((startIdx, endIdx)).T
    doneLogs = pd.DataFrame(columns = ['type','possible','stim','startInference','endInference', 'startNoference','endNoference','sortAnswer'])

    trials = []
    for row in limits:
        trials.append(lines[row[0]:row[1]])
   # return trials
    tmp = []
    for trial in trials:
        _type = [line.split(';')[2] for line in trial if "type" in line and '_' not in line][0]
        _possible = int([line.split(';')[5] for line in trial if "type" in line and '_' not in line][0])
        _stim = [line.split(';')[3] for line in trial if "type" in line and '_' not in line][0]
        _startInference =  [line.split(';')[0] for line in trial if "InferenceStarting" in line]
        _endInference =  [line.split(';')[0] for line in trial if "InferenceEnding" in line]     
        
        _startNoference =  [line.split(';')[0] for line in trial if "NoferenceStarting" in line]
        _endNoference =  [line.split(';')[0] for line in trial if "NoferenceEnding" in line]
        
        _sortAnswer =  str(trial[-1].split(';')[1])
       # doneLogs.append(pd.DataFrame(['_type','_possible','_stim','_startInference', '_endInference', '_sortAnswer']))
        doneLogs.loc[len(doneLogs)+1]=[_type,_possible,_stim,_startInference,_endInference,_startNoference,_endNoference, _sortAnswer]
        
    doneLogs[['startInference','endInference']] = doneLogs[['startInference','endInference']].applymap(lambda x: np.nan if len(x) == 0 else dt.datetime.strptime(x[0],'%H:%M:%S:%f'))
    doneLogs[['startNoference','endNoference']] = doneLogs[['startNoference','endNoference']].applymap(lambda x: np.nan if len(x) == 0 else dt.datetime.strptime(x[0],'%H:%M:%S:%f'))
    
    #indexes = np.where(doneLogs.loc[~doneLogs['_startInference'].isnull())
    indexes = doneLogs[~doneLogs['startInference'].isnull()].index.tolist()

    for i in indexes:
          doneLogs['startInference'].iloc[i-1] = doneLogs['startInference'].iloc[i-1].replace(year = file_time.year, month = file_time.month, day = file_time.day)
          doneLogs['endInference'].iloc[i-1] = doneLogs['endInference'].iloc[i-1].replace(year = file_time.year, month = file_time.month, day = file_time.day)
         
         
    indexes = doneLogs[~doneLogs['startNoference'].isnull()].index.tolist()
    for i in indexes:
          doneLogs['startNoference'].iloc[i-1] = doneLogs['startNoference'].iloc[i-1].replace(year = file_time.year, month = file_time.month, day = file_time.day)
          doneLogs['endNoference'].iloc[i-1] = doneLogs['endNoference'].iloc[i-1].replace(year = file_time.year, month = file_time.month, day = file_time.day)
    
    type_dict = {}
    types = doneLogs['type'].unique()
    types.sort()
    for idx,_type in enumerate(types):
        type_dict[_type] = idx
        
    doneLogs['int_type'] =  [0] * len(doneLogs.index)
    doneLogs['int_category'] =  [0] * len(doneLogs.index)
    doneLogs['int_inference'] =  [0] * len(doneLogs.index)
    doneLogs['accuracy'] =  [0] * len(doneLogs.index)
    for idx,row in doneLogs.iterrows():
        doneLogs['int_type'].loc[idx]= type_dict[row['type']]
        
        if(row['type'] in ['typeA', 'typeB', 'typeC', 'typeD']):
            doneLogs['int_category'].loc[idx] = 1
          
        if(row['type'] in ['typeB', 'typeC', 'typeI2', 'typeD']):
            doneLogs['int_inference'].loc[idx] = 1
        
        if((row['possible'] == 0) & (row['sortAnswer'] == 'answerBad\n')):
           # print('here')
            doneLogs['accuracy'].loc[idx] = 1
            
        if(row['possible'] == 1) and (row['sortAnswer'] == 'answerGood\n'):
            doneLogs['accuracy'].loc[idx] = 1
        
        if(row['type'] == 'typeI3') and (row['sortAnswer'] == 'cannot know'):
            doneLogs['sortAnswer'].loc[idx] = 1
        if(row['type'] != 'typeI3') and (row['sortAnswer'] == 'cannot know'):
            doneLogs['sortAnswer'].loc[idx] = 0
        
        if(row['sortAnswer'] == 'dont know'):
            doneLogs['sortAnswer'].loc[idx] = 0
            
        if('I' in row['type']):
            doneLogs['sortAnswer'].loc[idx] = int(doneLogs['sortAnswer'].loc[idx])
            #print('unt here')
            
    return doneLogs

def openEye(subjectID):
    
    _path = "/Users/ryszardcetnarski/Desktop/MasterForSync/Organized_Results/Eye/Subject_" + str(subjectID) + "/eye.csv"
    f = open(_path)
    lines = f.readlines()
    f.close()
    parsedLines = []
#Header lines to skip (and tobii recording untill unity started)
    skip_h = index_containing_substring(lines, 'Unity')
    my_index = []
    for idx, line in enumerate(lines):
        if(idx > skip_h and idx < len(lines) - 1000):
            splitLine = line.split("\t")
            epoch =int(splitLine[28].split('.')[0])
            epoch = epoch / 1000.0
            timestamp = dt.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %I:%M:%S.%f')
            my_index.append(timestamp)
            gaze = [int(splitLine[i]) for i in [4,5,11,12]]
            parsedLines.append(gaze)
            #parsedLines.extend(timestamp)
            
    parsedLines = np.array(parsedLines)
    dataFrame = pd.DataFrame(index = my_index, data = parsedLines, columns = ['left_x', 'left_y', 'right_x', 'right_y'])
    dataFrame.index = pd.to_datetime(dataFrame.index)
    return dataFrame
    
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1



def get_creation_time(path):
    p = subprocess.Popen(['stat', '-f%B', path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.wait():
        raise OSError(p.stderr.read().rstrip())
    else:
        return int(p.stdout.read())
#    