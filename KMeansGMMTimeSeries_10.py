#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:26:05 2022

Student ID: 29837043
File: KMeansGMMTimeSeries_10.py

This file contains the KMeans and GMM clustering to find the similarity between the account.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Import Config.py file
from Config_1 import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess_2 import *
# Import FuncLibVisual.py file
from FuncLibVisual_3 import *
# Import FuncLibModel.py file
from FuncLibModel_4 import *

from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore")

################### STEP 2: READ THE CLEANED DATA FILE ########################

# Calling ReadDataset to read the clean dataset obtained from DataPreProcessing_DataVisualization
dfbankdataset = ReadDataset(location_of_file, cleanfile)

# Taking a subset of dfbankdataset to data with only ACCOUNT_NO and TRANSACTION_AMOUNT attributes
data = dfbankdataset[['ACCOUNT_NO','TRANSACTION_DETAILS','TRANSACTION_AMOUNT']]
data['ACCOUNT_NO'] = data['ACCOUNT_NO'].apply(str)

######################## STEP 3: KMEANS ALGORITHM #############################

# Building KMeans algorithm to find the similarity between accounts by forming clusters [1] [2]

# Clear plots
plt.clf()
plt.cla()

# Create KMeans model with 10 clusters
kmeans = KMeans(n_clusters = 10)
kmeans.fit(data)
labels = kmeans.predict(data)

# Plot the scatter plot for KMeans
plt.scatter(x='ACCOUNT_NO', y='TRANSACTION_DETAILS', data=data, c=labels, s=20, cmap='viridis');
plt.scatter(x='ACCOUNT_NO', y='TRANSACTION_AMOUNT', data=data, c=labels, s=20, cmap='viridis');
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50, alpha=0.5);
plt.xlabel('ACCOUNT_NO')
plt.ylabel('TRANSACTION_DETAILS & TRANSACTION_AMOUNT')
plt.rcParams["figure.figsize"] = (8,4)
plt.show()

# Convert Centroids to dataframe
centroids = np.array(centers)
centroidsdf = pd.DataFrame({'Center_Act': (centroids[:, 0]), 'Center_TransDetailsAmount': centroids[:, 1]})
print(centroidsdf)

# Output:
#     Center_Act  Center_TransDetailsAmount
#0  4.090004e+11                  54.013304
#1  1.196478e+06                  59.044949
#2  4.090004e+11                  38.757399
#3  4.090005e+11                  44.564182
#4  4.090006e+11                  40.833486
#5  4.090004e+11                  72.620948
#6  4.090004e+11                  58.294118
#7  1.196428e+06                   0.000000
#8  1.196428e+06                   0.000000
#9  1.196428e+06                   0.000000

################# STEP 4: GET BEST CLUSTER BY ELBOW METHOD ####################

# List to store the metric value given different K values
inertia = []
#Range of the different values of K to analyse
K = range(2,11)

# Loop on different K values from 2 to 10
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data[['ACCOUNT_NO','TRANSACTION_DETAILS','TRANSACTION_AMOUNT']])
    inertia.append(km.inertia_)

# Plot the Elbow method
plt.plot(K, inertia)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.rcParams["figure.figsize"] = (8,6)
plt.show()

################# STEP 5: GET BEST CLUSTER BY GMM SILHOUETTE_SCORE METHOD ####################

# GMM with silhouette_score [3]

# Clear the lists
sscores.clear()
sresults.clear()

# Range of clusters to try (2 to 10)
K=range(2,11)

# Loop on different K values from 2 to 10
for k in K:
    # Set the model and its parameters
    model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
    # Fit the model
    model = model.fit(data)
    gmm_predict = model.predict(data)
    # Calculate Silhoutte Score and append to a list
    sscores.append(metrics.silhouette_score(data, gmm_predict, metric='euclidean'))

# Plot the resulting Silhouette scores on a graph
plt.figure(figsize=(8,4), dpi=300)
plt.plot(K, sscores, 'bo-', color='black')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Identify the number of clusters using Silhouette Score')
plt.show()

############################ STEP 6: GMM MODEL ################################

# Gaussian Mixture Model is build with 5 components are retrived from above step [4] [5]

# Create Scatter plot without clusters
plt.rcParams["figure.figsize"] = (14,8)
plt.scatter(data["ACCOUNT_NO"],data['TRANSACTION_DETAILS'],data["TRANSACTION_AMOUNT"])
plt.show()

# Create GMM model with 5 components derived from Silhouette Score
gmm = GaussianMixture(n_components=5)
gmm.fit(data)

# Predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['CLUSTER'] = labels
frame.columns = ['ACCOUNT_NO', 'TRANSACTION_DETAILS','TRANSACTION_AMOUNT', 'CLUSTER']

color=['blue','green','orange', 'black', 'red']
for k in range(0,4):
    data = frame[frame["CLUSTER"]==k]
    plt.scatter(data["ACCOUNT_NO"],data['TRANSACTION_DETAILS'], data["TRANSACTION_AMOUNT"],c=color[k])
plt.show()


###################### STEP 7: SAVE DATA PER CLUSTER ##########################

# Get the data of Cluster 0
dfcluster0 = frame.loc[frame['CLUSTER']==0]
# Save the cleaned dataset in xlsx file
dfcluster0.to_excel(cluster0data)

# Get the data of Cluster 0
dfcluster1 = frame.loc[frame['CLUSTER']==1]
# Save the cleaned dataset in xlsx file
dfcluster1.to_excel(cluster1data)

# Get the data of Cluster 0
dfcluster2 = frame.loc[frame['CLUSTER']==2]
# Save the cleaned dataset in xlsx file
dfcluster2.to_excel(cluster2data)

# Get the data of Cluster 0
dfcluster3 = frame.loc[frame['CLUSTER']==3]
# Save the cleaned dataset in xlsx file
dfcluster3.to_excel(cluster3data)

# Get the data of Cluster 0
dfcluster4 = frame.loc[frame['CLUSTER']==4]
# Save the cleaned dataset in xlsx file
dfcluster4.to_excel(cluster4data)


################################# References ##################################

# [1] https://compgenomr.github.io/book/clustering-grouping-samples-based-on-their-similarity.html
# [2] https://nzlul.medium.com/clustering-method-using-k-means-hierarchical-and-dbscan-using-python-5ca5721bbfc3
# [3] https://towardsdatascience.com/gmm-gaussian-mixture-models-how-to-successfully-use-it-to-cluster-your-data-891dc8ac058f
# [4] https://www.geeksforgeeks.org/gaussian-mixture-model/
# [5] https://machinelearningmastery.com/clustering-algorithms-with-python/

