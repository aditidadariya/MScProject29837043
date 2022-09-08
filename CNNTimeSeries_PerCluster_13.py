#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:24:28 2022

Student ID: 29837043
File: CNNTimeSeries_PerCluster_13.py

This file contains univariate time series forecasting in CNN model for each cluster formed.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot

import statistics
from statistics import mode

# Import Config.py file
from Config_1 import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess_2 import *
# Import FuncLibVisual.py file
from FuncLibVisual_3 import *
# Import FuncLibModel.py file
from FuncLibModel_4 import *

# Import mean_squared_error
from sklearn.metrics import mean_squared_error
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import StandardScaler to Standardize the data
#from sklearn.preprocessing import StandardScaler
# Import sqrt from math
from math import sqrt
# Import train_test_split function
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore")

################### STEP 2: READ THE CLEANED DATA FILE ########################

# Calling ReadDataset to read the clean dataset obtained from DataPreProcessing_DataVisualization
#dfbankdataset = ReadDataset(location_of_file, cleanfile)

# Calling ReadDataset funciton to read the data of all clusters
dfcluster0 = ReadDataset(location_of_file, cluster0data)
dfcluster1 = ReadDataset(location_of_file, cluster1data)
dfcluster2 = ReadDataset(location_of_file, cluster2data)
dfcluster3 = ReadDataset(location_of_file, cluster3data)
dfcluster4 = ReadDataset(location_of_file, cluster4data)

# Save all the dataframes into a list
ClusterData = [dfcluster0, dfcluster1, dfcluster2, dfcluster3, dfcluster4]

#################### STEP 3: BUILD CNN MODEL PER CLUSTER  #####################

start_time = datetime.now()
#print("Final LSTM model start_time is - ", start_time)

# Cleat the list
rmselist.clear()

best_units = 182
best_epoch = 50
best_learning_rate = 0.01

# ============= 3.1. Evaluating CNN Model for each Cluster ================

# Evaluate LSTM model for each cluster with the best hyperparameters found previuously
# Creating a forecast and getting the Root Mean Squared Error
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata["TRANSACTION_AMOUNT"] = dfeach["TRANSACTION_AMOUNT"]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
     
    # ================== 3.2. Split the data in train and test  ===================

    # Calling SplitTrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    # Reshaping the Input Data as per Conv2D layers
    trainConv2D, testConv2D = ReshapeCNNData(trainingset, testset)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainConv2D.shape, testConv2D.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 3.3. Create CNN Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainConv2D, testConv2D, n_input)
    # Calling FinalCNNModel function to create the model object for CNN
    cnnmodel = FinalCNNModel(trainConv2D, n_input, n_features, best_units)
    
    # ================ 3.4. Compile and Fit the CNN Model ========================
    
    # Calling FinalCompileFitModel function to fit the CNN model
    cnnmodelfit = FinalCompileFitModel(cnnmodel, traingenerator, testgenerator, best_epoch, best_learning_rate)
     
    # ===================== 3.5. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ===================== 3.6. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ======================== 3.7. Plot the future ===============================
    
    # Plot the Test Predictions
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, "Cluster"+str(index))
    
    # Printing the RMSE for Account
    rmseline = {'CLUSTER': "Cluster"+str(index), 'CNN_TESTRMSE': testrmse, 'CNN_FUTURERMSE': futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# Show all the plots
plt.show()
# Printing the final RMSE against each account
df_finalrmse = pd.DataFrame(rmselist)
print(df_finalrmse)

# Plot RMSE for each Cluster in barplot
df_finalrmse[['CNN_TESTRMSE','CNN_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Final CNN model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final CNN model for all accounts is {}".format(totaltime))

#Output:
#    CLUSTER  CNN_TESTRMSE  CNN_FUTURERMSE
#0  Cluster0      0.042754        0.042754
#1  Cluster1      0.000107        0.000107
#2  Cluster2      0.148843        0.148843
#3  Cluster3      0.000940        0.000940
#4  Cluster4      0.005505        0.005505

# Total time to run the Final CNN model for all accounts is 5:22:59.049160

