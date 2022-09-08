#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:32:08 2022

Student ID: 29837043
File: LSTMTimeSeries_PerCluster_12.py

This file contains univariate time series forecasting in LSTM model for each cluster formed.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot

from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
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
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout #https://www.codegrepper.com/code-examples/whatever/NameError%3A+name+%27Dropout%27+is+not+defined
from keras.layers import Flatten
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from numpy import hstack

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

################### STEP 3: BUILD THE LSTM MODEL PER CLUSTER #######################

start_time = datetime.now()
#print("Final LSTM model start_time is - ", start_time)

# Cleat the list
rmselist.clear()

best_units = 182
best_epoch = 20
best_learning_rate = 0.01

# ============= 3.1. Evaluating LSTM Model for each Cluster ================

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
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 3.3. Create LSTM Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainingset, testset, n_input)
    # Calling FinalLSTMModel function to create the model object for LSTM
    lstmmodel = FinalLSTMModel(trainingset, n_input, n_features, best_units)
    
    # ================ 3.4. Compile and Fit the LSTM Model ========================
    
    # Calling FinalCompileFitModel function to fit the LSTM model
    lstmmodelfit = FinalCompileFitModel(lstmmodel, traingenerator, testgenerator, best_epoch, best_learning_rate)
    
    # ===================== 3.6. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ===================== 3.7. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ======================== 3.8. Plot the future ===============================
    
    # Plot the Test Predictions
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, "Cluster"+str(index))
    
    # Printing the RMSE for Account
    rmseline = {'CLUSTER': "Cluster"+str(index), 'LSTM_TESTRMSE': testrmse, 'LSTM_FUTURERMSE': futurermse}
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
df_finalrmse[['LSTM_TESTRMSE','LSTM_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Final LSTM model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final LSTM model for all clusters is {}".format(totaltime))


#Output:
#    CLUSTER  LSTM_TESTRMSE  LSTM_FUTURERMSE
#0  Cluster0       0.083213         0.076200
#1  Cluster1       0.000645         0.000645
#2  Cluster2       0.037557         0.037557
#3  Cluster3       0.000106         0.000106
#4  Cluster4       0.020466         0.020466

# Total time to run the Final LSTM model for all clusters is 1:40:52.966923

