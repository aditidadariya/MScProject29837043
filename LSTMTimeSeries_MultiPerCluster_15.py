#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:04:44 2022

Student ID: 29837043
File: LSTMTimeSeries_MultiPerCluster_15.py

This file contains multivariate time series forecasting in LSTM model for each cluster formed.

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
# Import scikit-learn metrics 
from sklearn import metrics
from numpy import array
from numpy import hstack
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
from keras.layers import Dropout
from keras.layers import Flatten
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore")

################### STEP 2: READ THE CLEANED DATA FILE ########################

# Calling ReadDataset funciton to read the data of all clusters
dfcluster0 = ReadDataset(location_of_file, cluster0data)
dfcluster1 = ReadDataset(location_of_file, cluster1data)
dfcluster2 = ReadDataset(location_of_file, cluster2data)
dfcluster3 = ReadDataset(location_of_file, cluster3data)
dfcluster4 = ReadDataset(location_of_file, cluster4data)

# Save all the dataframes into a list
ClusterData = [dfcluster0, dfcluster1, dfcluster2, dfcluster3, dfcluster4]

################### STEP 3: BUILD LSTM MULTIVARIATE MODEL PER CLUSTER #######################

start_time = datetime.now()
#print("Multivariate LSTM model start_time is - ", start_time)

rmselist.clear()

best_units = 182
best_epoch = 20
best_learning_rate = 0.01

# ============= 3.1. Evaluating LSTM Multivariate Model for each Cluster ================

# Evaluate LSTM model for each cluster with the best hyperparameters found previuously
# Creating a forecast and getting the Root Mean Squared Error
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]] = dfeach[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')    # [1]
    
    # ================== 3.2. Split the data in train and test  ===================

    # Calling TrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    #Output:
    # For Cluster1 Training set is (41520, 2) and Test set is (17795, 2)
    
    # ================== 3.3. Define the train and test data ===================
    
    # Get the individual attributes of train data [2]
    train_in_seq1 = df_train['TRANSACTION_DETAILS'].values
    train_in_seq2 = df_train['TRANSACTION_AMOUNT'].values
    # reshape train series
    train_in_seq1 = train_in_seq1.reshape((len(train_in_seq1), 1))
    train_in_seq2 = train_in_seq2.reshape((len(train_in_seq2), 1))
    # Get the individual attributes of test data
    test_in_seq1 = df_test['TRANSACTION_DETAILS'].values
    test_in_seq2 = df_test['TRANSACTION_AMOUNT'].values
    # Reshape test series
    test_in_seq1 = test_in_seq1.reshape((len(test_in_seq1), 1))
    test_in_seq2 = test_in_seq2.reshape((len(test_in_seq2), 1))
    
    print("Reshaped train and test datasets {} {} {} {} ".format(train_in_seq1.shape, train_in_seq2.shape, test_in_seq1.shape, test_in_seq2.shape))
    
    # horizontally stack columns
    train_dataset = hstack((train_in_seq1, train_in_seq2))
    test_dataset = hstack((test_in_seq1, test_in_seq2))
    
    # define generator parameters
    n_features = train_dataset.shape[1]
    n_input = 2
    
    # Output:
    # Reshaped train and test datasets (13226, 1) (13226, 1) (5669, 1) (5669, 1) 
    
    # ========================= 3.4. Create LSTM Model ============================
    
    # Define generator [3] [4]
    generator = TimeseriesGenerator(train_dataset, train_dataset, length=n_input, batch_size=8)
    
    # Calling FinalLSTMModel function to create the model object for LSTM
    lstmmodel = FinalLSTMModel(train_dataset, n_input, n_features, best_units)
    
    # ================ 3.5. Compile and Fit the LSTM Model ========================
    
    # Calling MultiCompileFitModel function to fit the LSTM model
    lstmmodelfit = MultiCompileFitModel(lstmmodel, generator, best_epoch, best_learning_rate)
    
    # ===================== 3.6. Predict the test data ============================

    # Calling MultiTestPrediction function to create prediction on test data
    td_testrmse, ta_testrmse, df_testpredictions = MultiTestPrediction(train_dataset, df_test, n_input, n_features, lstmmodelfit)
    
    # ===================== 3.7. Predict the future ===============================
        
    # Calling MultiFuturePrediction function to create future predictios using testset
    td_futurermse, ta_futurermse, df_futurepredictions = MultiFuturePrediction(train_dataset, df_test, n_input, n_features, lstmmodelfit)
    
    # ======================== 3.8. Plot the future ===============================
    
    # Plot the Test Predictions
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, "Cluster"+str(index))
    
    # Store the RMSE for Clusters
    rmseline = {"CLUSTER": "Cluster"+str(index), "TD_TESTRMSE": td_testrmse, "TD_FUTURERMSE": td_futurermse[0], "TA_TESTRMSE": ta_testrmse, "TA_FUTURERMSE": ta_futurermse[0]}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# Show all the plots
plt.show()
# Printing the final RMSE against each cluster
df_finalrmse = pd.DataFrame(rmselist)
print("LSTM Multivariate Model: Clusters with its RMSE")
print(df_finalrmse)

# Plot RMSE for each Cluster in barplot
df_finalrmse[['TD_TESTRMSE','TD_FUTURERMSE','TA_TESTRMSE','TA_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Final LSTM model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final LSTM multivairate model for all clusters is {}".format(totaltime))


#Output:
#LSTM Multivariate Model: Clusters with its RMSE
#    CLUSTER  TD_TESTRMSE  TD_FUTURERMSE  TA_TESTRMSE  TA_FUTURERMSE
#0  Cluster0    14.249114      54.997581     0.204589       0.204286
#1  Cluster1    18.279041      44.751015     0.080448       0.053527
#2  Cluster2    19.882582      49.493748     0.315159       0.190826
#3  Cluster3    10.178728      37.909245     0.006071       0.001346
#4  Cluster4     8.644634      45.290577     0.437827       0.439380

#Total time to run the Final LSTM multivairate model for all clusters is 1:12:35.475609


################################# References ##################################

# [1] https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr
# [2] https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# [3] https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# [4] https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/



