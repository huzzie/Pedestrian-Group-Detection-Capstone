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
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# data preprocessing
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

# normalization
def normalization(df):
    columns = ['xcenter', 'ycenter', 'bbox0', 'bbox1', 'bbox2', 'bbox3']
    scale = MinMaxScaler()
    for i in columns:
        df[i] = scale.fit_transform(df[i].values.reshape(-1, 1))
    return df

# time series Kmeans
def dtw(df, k):
    new_df = df[['xcenter', 'ycenter']]
    model = TimeSeriesKMeans(n_clusters = k, metric = 'dtw', max_iter = 5)
    model.fit(new_df)
    res = model.fit_predict(new_df)
    return res

# find the centtroids of dtw
def db_centroids(df, k):
    dtw_res = dtw(df, k)
    df2 = df.copy()
    df2['dtw_cluster'] = dtw_res
    df2 = df2.groupby(['dtw_cluster']).mean()
    return df2[['xcenter', 'ycenter']]

   
# evaluate dtw with the sihouette score 
def dtw_eval(df, k):
    dtw_res = dtw(df, k)
    df2 = df.copy()
    df2['dtw_cluster'] = dtw_res
    silvalue = silhouette_score(df2[['xcenter', 'ycenter']], df2['dtw_cluster'])
    print('silhouette coefficient', silvalue)
    

# plot the trajectories by cluster
def plot_dtw(df, k):
    cluster = dtw(df, k)
    df2 = df[['xcenter', 'ycenter']]
    df2['cluster'] = cluster
    sns.scatterplot(x = 'xcenter', y = 'ycenter', hue = 'cluster', style = 'cluster', data = df2, legend = None)
    plt.show()



    
    
    
