# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:30:13 2015

@author: ryszardcetnarski
"""
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as sklearnPCA
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def myPCA(time_series, known_categories):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
#Determine how many PC are expected. This parameter does not changbaseline ie the PCA just limits the output to the n components    
    sklearn_pca = sklearnPCA(n_components=3)
    sklearn_pca.fit(time_series)
#Apply the PCA o the data
    sklearn_transf = sklearn_pca.transform(time_series)
#Trace the results of the PCA back to the original dat. Altough it seems an unnecessary step as the sklearn_transf returning the components values maintains the same order as in the original data
    #scores = sklearn_pca.inverse_transform(sklearn_transf)

    fig.suptitle('Gaze X position PCA', fontsize = 18)
    
    ax2.plot(sklearn_pca.explained_variance_ratio_)
    
    tick_locs = [0,1,2]
    tick_lbls = ['1','2','3']
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_lbls)    
    
    ax2.set_xlabel('Component order')
    ax2.set_ylabel('Explained variance ratio')
    plt.show()
    
#Plot the results of the PCA in 2D (i.e. first agains the second PC).
    for i in range(0, len(sklearn_transf[:,0])):
        if(known_categories[i] == 1):
            ax.plot(sklearn_transf[i,0],sklearn_transf[i,1], 'bo', alpha = 0.6)
        else:
            ax.plot(sklearn_transf[i,0],sklearn_transf[i,1], 'ro', alpha = 0.6)

    ax.set_xlabel('1st principal component score')
    ax.set_ylabel('2nd principal component score')

    x, y, z = sklearn_transf[:,0], sklearn_transf[:,1], sklearn_transf[:,2]
    components = np.vstack((x,y,z)).T    
    #outliers_idx = remove_outliers_bis(components, 2) 
   # outliers = components[outliers_idx]
    ax.legend()
    #ax.plot(outliers[:, 0], outliers[:,1], 'ro')

    plt.ion()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
#Plot the results of the PCA in 3d, i.e. 1stPC x 2ndPC x 3rdPC    
    for i in range(0, len(components)):
        if(known_categories[i] == 1):
            ax.scatter(components[i,0],components[i,1],components[i,2], color = 'b', alpha = 0.2)
        else:
            ax.scatter(components[i,0],components[i,1],components[i,2], color = 'r', alpha = 0.2) 
    
    #ax.scatter(x, y, z, 'b')

    #ax.scatter(outliers[:, 0], outliers[:,1], outliers[:,2], color = 'r')

    ax.set_xlabel('1st pc')
    ax.set_ylabel('2nd pc')
    ax.set_zlabel('3rd pc')    
    #return sklearn_pca.components_, sklearn_pca.explained_variance_ratio_, sklearn_pca.mean_, sklearn_pca.n_components_, sklearn_pca.noise_variance_, sklearn_transf, scores#, sklearn_transf, scores    
        
    #components = np.delete(components, (outliers_idx), axis=0)
    return components
    
    
def remove_outliers_bis(arr, k):
    mask = np.ones((arr.shape[0],), dtype=np.bool)
    mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask[mask] &= np.abs((col[mask] - mu[j]) / sigma[j]) < k
    return np.where(mask == False)[0]
    