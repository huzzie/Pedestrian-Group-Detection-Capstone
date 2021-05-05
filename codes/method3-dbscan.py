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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import itertools
# =============================================================================
#  Data Preprocessing
# =============================================================================
def normalization(df):
    columns = ['xcenter', 'ycenter', 'bbox0', 'bbox1', 'bbox2', 'bbox3']
    scale = MinMaxScaler()
    for i in columns:
        df[i] = scale.fit_transform(df[i].values.reshape(-1, 1))
    return df

# =============================================================================
# Elbow method
# =============================================================================
def dbscan_elbow(k, df):
    nn = NearestNeighbors(n_neighbors = k).fit(df[['xcenter', 'ycenter']])
    distances, index = nn.kneighbors(df[['xcenter', 'ycenter']])
    distances = np.sort(distances, axis = 0)
    distances = distances[:, 1]
    #return distances
    plt.plot(distances)
    plt.show()


# evluation epsilons 
def dbscan_eval(epsilon, df, min_):
    dbs = DBSCAN(eps = epsilon, min_samples = min_).fit(df[['xcenter', 'ycenter']])
    labels = dbs.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noises = list(labels).count(-1)
    silvalue = silhouette_score(df[['xcenter', 'ycenter']], labels)
    print('number of clusters', n_clusters)
    print('number of noises', noises)
    print('silhouette coefficient', silvalue)
    

# elbow plot for multiple parrameters
def evaluation_param(df):
    eps_ = np.arange(0.005, 0.011, 0.001)
    mins = np.arange(2, 5) # if min sample is greater than 5, it returns error
    parameters = list(itertools.product(eps_, mins))
    clusters = []
    sil_score = []
    eps_val = []
    mins_val = []

    for p in parameters:
        dbscan_clusters = DBSCAN(eps = p[0], min_samples = p[1]).fit(df[['xcenter', 'ycenter']])
        eps_val.append(p[0])
        mins_val.append(p[1])
        clusters.append(len(np.unique(dbscan_clusters.labels_)))
        sil_score.append(silhouette_score(df[['xcenter', 'ycenter']], dbscan_clusters.labels_))
        df_res = pd.DataFrame({'cluster': dbscan_clusters,
                               'eps': eps_val,
                               'min_val': mins_val,
                               'sil_val': sil_score})
    return df_res

# reference : https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31#:~:text=DBSCAN%20Cluster%20Evaluation,each%20point%20in%20other%20clusters.


# =============================================================================
# Using DBSCAN algorithim
# =============================================================================

def dbscan_cluster(df, distances):

    # model fit using DBSCAN
    model = DBSCAN(eps=distances, min_samples = 2, metric = 'haversine')
    model.fit(df[['xcenter', 'ycenter']])
    samples = np.zeros_like(model.labels_, dtype = bool)
    samples[model.core_sample_indices_] = True
    labels = model.labels_
    df['cluster'] = model.labels_.tolist()
    
    # cluster by track id
    get_id = list(df['track_id'].unique())
    random_cluster = get_id[random.randint(0, len(get_id)-1)]
    rand_cluster = set()
    for i in range(len(df)-1):
        if df['track_id'].iloc[i] == random_cluster:
            rand_cluster.add(df['cluster'].iloc[i])
    
    # find clusters within the group
    groups = set()
    
    for cluster in rand_cluster:
        if cluster != -1:
            group_incluster = df.loc[df['cluster'] == cluster, 'track_id']
            for i in range(len(group_incluster)):
                group = group_incluster.iloc[i]
                if group != random_cluster:
                    groups.add(group)
    return groups


# =============================================================================
# Clustered dbscan evaluation
# =============================================================================
# evaluate the dbscan_cluster
def dbscan_cluster_eval(df, distances) :
    dbscan_result = dbscan_cluster(df, distances)
    df2 = df[df['track_id'].isin(list(dbscan_result))]
    dbs = DBSCAN(eps = distances, min_samples = 2).fit(df2[['xcenter', 'ycenter']])
    labels = dbs.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noises = list(labels).count(-1)
    silvalue = silhouette_score(df2[['xcenter', 'ycenter']], labels)
    print('number of clusters', n_clusters)
    print('number of noises', noises)
    print('silhouette coefficient', silvalue)


# =============================================================================
# Find centroids of DBSCAN 
# =============================================================================
def dbscan_centroids(df):
    df2 = df.groupby(['track_id']).mean()
    return df2[['xcenter', 'ycenter']]



# =============================================================================
# Plot DBSCAN trajectories
# =============================================================================
def dbscan_track_plot(df):
    sns.scatterplot(x = 'xcenter', y = 'ycenter', hue="track_id", palette="ch:r=-.2,d=.3_r", data = df, legend = False)
    sns.scatterplot(x = 'xcenter', y = 'ycenter', color="red", data = dbscan_centroids(df), legend = False)
    plt.show()

