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
import pygal
from sklearn.cluster import DBSCAN
from IPython.display import display, HTML
import random
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

# trajectories for starting points
def centroid_starting_point(df):
    xmin = df.groupby(['track_id'], sort = False)['xcenter'].min()
    ymin = df.groupby(['track_id'], sort = False)['ycenter'].min()
    dataset = [xmin, ymin]
    dataset =np.asarray(dataset)
    dataset = np.transpose(dataset)
    return dataset
    
# trajectories for ending points
def centroid_ending_point(df):
    xmax = df.groupby(['track_id'], sort = False)['xcenter'].max()
    ymax = df.groupby(['track_id'], sort = False)['ycenter'].max()
    dataset = [xmax, ymax]
    dataset =np.asarray(dataset)
    dataset = np.transpose(dataset)
    return dataset


def dataset(df):
    xmin = df['bbox0']
    ymin = df['bbox1'] 
    xmax = df['bbox2'] 
    ymax = df['bbox3'] 
    
    dataset = [xmax - xmin, ymax - ymin]
    dataset = np.asarray(dataset)
    dataset = np.transpose(dataset)
    return dataset

# =============================================================================
# Hypothesis 1
# tracking the framees per second of each approach / run the number of seconds
#  Use Elbow method to choose the number of k clusters
# =============================================================================

def elbow_kmeans(df):
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
        km.fit(df)
        sse.append(km.inertia_)
    return sse

# =============================================================================
# K Means
# =============================================================================

def kmeans(k, df):
    km = KMeans(n_clusters = k)
    labels = km.fit_predict(df[['xcenter', 'ycenter']])
    centroids = km.cluster_centers_
    return centroids

# =============================================================================
# K Means evaluation using silhouette score
# =============================================================================
def KMeans_silhouette(k, df):
    clusters = KMeans(n_clusters = k, random_state =0)
    clusters_labels = clusters.fit_predict(df[['xcenter', 'ycenter']])
    res = silhouette_score(df[['xcenter', 'ycenter']], clusters_labels)
    return res

# =============================================================================
# Plot
# =============================================================================
def plot_km_centroids(df, k):
    km = KMeans(n_clusters = k)
    labels = km.fit_predict(df[['xcenter', 'ycenter']])
    centroids = km.cluster_centers_
    res = pd.DataFrame(centroids, columns = ['xcenter', 'ycenter'] )
    #sns.scatterplot(x = 'xcenter', y = 'ycenter', hue="track_id", palette="ch:r=-.2,d=.3_r", data = df, legend = False)
    sns.scatterplot(x = 'xcenter', y = 'ycenter', color = 'red',  data = res, legend = False)
    plt.show()

# =============================================================================
# Result - path 
# =============================================================================

def kmeans_track_plot(df, CLUSTER):
    data = dataset(df)
    output = kmeans(data, k = CLUSTER)
    kmeans_df = pd.DataFrame({'xvalue': output[:, 0], 'yvalue': output[ :, 1]})
    #sns.scatterplot(x = 'xcenter', y = 'ycenter', hue="track_id", palette="ch:r=-.2,d=.3_r", data = df, legend = False)
    sns.scatterplot(x = 'xvalue', y = 'yvalue', color = 'red', data = kmeans_df, legend = False)
    plt.plot()

# =============================================================================
# Result 2 - centroid starting point
# =============================================================================

def kmeans_starting_plot(df, CLUSTER):
    data = centroid_starting_point(df)
    output = kmeans(data, k = CLUSTER)
    kmeans_df = pd.DataFrame({'xvalue': output[:, 0], 'yvalue': output[ :, 1]})
    # plot 
    sns.scatterplot(x = 'xcenter', y = 'ycenter', hue="track_id", palette="ch:r=-.2,d=.3_r", data = df, legend = False)
    sns.scatterplot(x = 'xvalue', y = 'yvalue', color = 'red', data = kmeans_df, legend = False)
    plt.plot()

# =============================================================================
# Result 3 - centroid starting and ending points
# =============================================================================

def kmeans_centroid_plot(df, CLUSTER):
    point_start = centroid_starting_point(df)
    point_end = centroid_ending_point(df)
    output1 = kmeans(point_start, k = CLUSTER)
    output2 = kmeans(point_end, k = CLUSTER)
    kmeans_df1 = pd.DataFrame({'xvalue': output1[:, 0], 'yvalue': output1[ :, 1]})
    kmeans_df2 = pd.DataFrame({'xvalue': output2[:, 0], 'yvalue': output2[ :, 1]})
    # plot 
    sns.scatterplot(x = 'xcenter', y = 'ycenter', hue="track_id", palette="ch:r=-.2,d=.3_r", data = df, legend = False)
    sns.scatterplot(x = 'xvalue', y = 'yvalue', color = 'red', data = kmeans_df1, legend = False)
    sns.scatterplot(x = 'xvalue', y = 'yvalue', color = 'blue', data = kmeans_df2, legend = False)
    plt.plot()

# =============================================================================
# CV2 (video and starting and ending point )
# =============================================================================
def kmeans_centroid_plot(df, CLUSTER):
    point_start = centroid_starting_point(df)
    point_end = centroid_ending_point(df)
    output1 = kmeans(point_start, k = CLUSTER)
    output2 = kmeans(point_end, k = CLUSTER)
    kmeans_df1 = pd.DataFrame({'xvalue': output1[:, 0], 'yvalue': output1[ :, 1]})
    kmeans_df2 = pd.DataFrame({'xvalue2': output2[:, 0], 'yvalue2': output2[ :, 1]})
    kmeans_df = pd.concat([kmeans_df1, kmeans_df2], axis = 1)
    return kmeans_df
