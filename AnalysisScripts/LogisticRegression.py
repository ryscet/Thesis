# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:56:40 2015

@author: ryszardcetnarski
"""
legend= 'accuracy, selectedNodeIndex, trialCounter, condition, ar_State, cl_State, ar_intervention, cl_intervention, pathIndex, trial_second, timeOfStep, resetTime, optimalNodeToEnd, opt_1, csq_1, opt_2, csq_2, opt_3, csq_3'

import pandas as pd
import statsmodels.api as sm
#import pylab as pl
import numpy as np
#import BasicAnalysis as ba



def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
 
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
 
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
 
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

#events = ba.GetEvents()
##Use concat to combine a list or dict of dataframes with the same dimensions. Uber useful
#events = pd.concat(events)
#
#events.drop(events.columns[9:19],inplace=True, axis = 1)
#
##print(events.head())
#
##print (events.describe())
##VERY NICE THE CROSSTABS!!! -  frequency table usable for chi square testing
##print(pd.crosstab(events['accuracy'], events['cl_intervention'], rownames=['accuracy']))
#
##events.hist()
##pl.show()
#
#
## dummify rank
#dummy_difficulty = pd.get_dummies(events['pathIndex'], prefix='pathIndex')
#dummy_condition = pd.get_dummies(events['condition'], prefix='condition')
## Do not include all the dummy categories because it creates some statistical problem (multicolinearity), which means that n-1 categories is enough to infer all values in n categories as they are mutually exclusive
#cols_to_keep = ['accuracy', 'ar_State', 'cl_State', 'ar_intervention','cl_intervention', 'guiTimer']
#
#data = events[cols_to_keep].join(dummy_difficulty.ix[:, 'pathIndex_1.0':])
#data = data.join(dummy_condition.ix[:, 'condition_1.0':])
#
## manually add the intercept
#data['intercept'] = 1.0
#
#train_cols =['ar_State', 'cl_State', 'pathIndex_1.0', 'pathIndex_2.0']
#
##subData = data[train_cols]
def logReg(dependent, independent):

    #First give the variable the model will try to predict (the dependent) then the predictors
    #logit = sm.Logit(data['accuracy'], data[train_cols])
    logit = sm.Logit(dependent,independent)
    
    # fit the model
    result = logit.fit()
    
    print(result.summary())
    print('\n'+"Confidence Intervals" + '\n')
    print(result.conf_int())
    print('\n'+"Odds ratio" + '\n')
    
    print(np.exp(result.params))
    
    print('\n'+"-------" + '\n')
    
#    # odds ratios and 95% CI
#    params = result.params
#    conf = result.conf_int()
#    conf['OR'] = params
#    conf.columns = ['2.5%', '97.5%', 'OR']
#    print (np.exp(conf))
    
    
    
    # instead of generating all possible values of GRE and GPA, we're going
    # to use an evenly spaced range of 10 values from the min to the max 
    #gres = np.linspace(independent.min(), independent.max(), 10)
    #gpas = np.linspace(data['cl_State'].min(), data['cl_State'].max(), 10)
    
    # enumerate all possibilities
    #combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3], [1.]]))
    
    # recreate the dummy variables
    #combos.columns = ['ar_State', 'cl_State', 'pathIndex', 'intercept' ]
    #dummy_ranks_1 = pd.get_dummies(combos['pathIndex'], prefix='pathIndex')
    #dummy_ranks_1.columns = ['pathIndex_0.0', 'pathIndex_1.0', 'pathIndex_2.0']
    #
    ## keep only what we need for making predictions
    #cols_to_keep = ['ar_State', 'cl_State', 'pathIndex', 'intercept' ]
    #combos = combos[cols_to_keep].join(dummy_ranks_1.ix[:, 'pathIndex_1':])
    #
    #combos['accuracy_predict'] = result.predict(combos[['ar_State', 'cl_State', 'pathIndex_1.0', 'pathIndex_2.0']])
    #print('\n'+"-------" + '\n')
    
#    print (combos.head())

#
#def isolate_and_plot(variable):
#    # isolate gre and class rank
#    grouped = pd.pivot_table(combos, values=['accuracy_predict'], index=[variable, 'pathIndex'],
#                            aggfunc=np.mean)
#    
#    # make a plot
#    colors = 'rbgyrbgy'
#    for col in combos.pathIndex.unique():
#        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
#        pl.plot(plt_data.index.get_level_values(0), plt_data['accuracy_predict'],
#                color=colors[int(col)])
# 
#    pl.xlabel(variable)
#    pl.ylabel("P(acc=1)")
#    pl.legend(['easy','med','hard'], loc='upper left', title='diff')
#    pl.title("Prob(acc=1) isolating " + variable + " and diff")
#    pl.show()
# 
#isolate_and_plot('ar_State')
##isolate_and_plot('cl_State')
#
#print('end')