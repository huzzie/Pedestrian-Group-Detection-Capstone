import os
import pandas as pd
import glob
import csv
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import numpy as np
import time
from sklearn.metrics import silhouette_samples, silhouette_score

# =============================================================================
# Data Preprocessing
# =============================================================================
def data_preprocessing(dataset):
    df = pd.read_csv(dataset, index_col = 0)
    df['bbox'] = df['bbox'].str.replace('[', '')
    df['bbox'] = df['bbox'].str.replace(']', ' ')
    df['bbox_split'] = df['bbox'].str.split()
    df['bbox_split'] = [x for x in df['bbox_split'] if x]
    df2 = df.bbox_split.apply(pd.Series)
    renames = {0: 'bbox0', 1: 'bbox1', 2: 'bbox2', 3: 'bbox3'}
    df2.rename(columns = renames, inplace = True )
    df  = pd.concat([df, df2], axis = 1)
    
    # convert object to strings
    df['bbox0'] = df['bbox0'].astype(float)
    df['bbox1'] = df['bbox1'].astype(float)
    df['bbox2'] = df['bbox2'].astype(float)
    df['bbox3'] = df['bbox3'].astype(float)
    # find the width
    df['b_w'] = df['bbox2'] - df['bbox0']
    # find the height
    df['b_h'] = df['bbox3'] - df['bbox1']
    return df

# =============================================================================
#  K means for hypothesis 3 - kmeans for centroids of moving objects
# Extended k-means algorithm group trajectories featuring similar motion patterns
# =============================================================================

class KMeans3:
    def __init__(self, k = 4, tol= 0.001, max_iter= 3):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break
            
 
    def update(self, new_data, delta):
        for featureset in new_data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]

            if min(distances) < delta:
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            else:
                self.centroids[self.k] = featureset
                self.classifications[self.k] = []
                self.classifications[self.k].append(featureset)
                self.k = self.k + 1
                
# reference: https://gist.github.com/spikar/5a877fb1bca34444bcbf016dc180821d

# =============================================================================
#    Evaluation             
# =============================================================================

def plot_centroids(df):
    hypothesis3 = KMeans3()
    X = np.array(df[['xcenter', 'ycenter']])  
    hypothesis3.fit(X) 
   # sns.scatterplot(x = 'xcenter', y = 'ycenter', hue = 'track_id', style = 'track_id', data = df, legend = None)
    for centroid in hypothesis3.centroids:
        plt.scatter(hypothesis3.centroids[centroid][0], hypothesis3.centroids[centroid][1], color="red")
    plt.show()


# return the centroids values
def evaluation3(df):
    hypothesis3 = KMeans3()
    X = np.array(df[['xcenter', 'ycenter']])  
    hypothesis3.fit(X) 
    #res = hypothesis3.centroids
    xcenter = []
    ycenter = []
    for centroid in hypothesis3.centroids:
        xcenter.append(hypothesis3.centroids[centroid][0])
        ycenter.append(hypothesis3.centroids[centroid][1])
    res = pd.DataFrame({'xcenter': xcenter, 'ycenter': ycenter}) 
    return res

        






                