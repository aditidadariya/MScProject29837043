#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:24:28 2022

Student ID: 29837043
File: CNNTimeSeries_PerAccount_9.py

This file contains univariate time series forecasting in CNN model for each account.

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
dfbankdataset = ReadDataset(location_of_file, cleanfile)

###################### STEP 3: BUILD BASIC CNN MODEL  ########################

start_time = datetime.now()
#print("Basic CNN model start_time is - ", start_time)

# =============== 3.1. Group the dataset for based ACCOUNT_NO =================

# Grouping the ACCOUNT_NO based on maximum number of records for each account in the dfbankdataset dataframe 
actcount = dfbankdataset.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()

# Clear the list 
rmselist.clear()

# ============= 3.2. Evaluating CNN Model for each ACCOUNT_NO ================

# Building the basic model for each account, creating a forecast and getting the Root Mean Squared Error
for ind, eachact in actcount.items():

    # Getting TRANSACTION_AMOUNT attribute values for each ACCOUNT_NO into another dataframe
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M')
    
    # ================== 3.3. Split the data in train and test  ===================

    # Calling TrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    # Reshaping the Input Data as per Conv2D layers
    trainConv2D, testConv2D = ReshapeCNNData(trainingset, testset)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainConv2D.shape, testConv2D.shape))
    
    # Output:
    # For Account 1196428 Training set is (34145, 1, 1) and Test set is (14634, 1, 1)
    
    # ========================= 3.4. Create CNN Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainConv2D, testConv2D, n_input)
    # Create Conv2D Model
    cnnmodel = CreateCNNModel(trainConv2D, n_input, n_features)
    
    #CNN layers
    #Model: "sequential"
    #_________________________________________________________________
    # Layer (type)                Output Shape              Param #   
    #=================================================================
    # conv2d (Conv2D)             (1, 10, 1, 64)            640       
    #                                                                 
    # max_pooling2d (MaxPooling2D  (1, 10, 1, 64)           0         
    # )                                                               
    #                                                                 
    # conv2d_1 (Conv2D)           (1, 10, 1, 32)            18464     
    #                                                                 
    # max_pooling2d_1 (MaxPooling  (1, 10, 1, 32)           0         
    # 2D)                                                             
    #                                                                 
    # flatten (Flatten)           (1, 320)                  0         
    #                                                                 
    # dropout (Dropout)           (1, 320)                  0         
    #                                                                 
    # dense (Dense)               (1, 50)                   16050     
    #                                                                 
    # dense_1 (Dense)             (1, 1)                    51        
    #                                                                 
    #=================================================================
    #Total params: 35,205
    #Trainable params: 35,205
    #Non-trainable params: 0
    #_________________________________________________________________


    # ================ 3.5. Compile and Fit the CNN Model ========================
    
    # Calling CompileFitModel function to fit the CNN model
    cnnmodelfit = CompileFitModel(cnnmodel, traingenerator, testgenerator)
    
    # ================== 3.6. Plot the CNN Model losses ==========================
    
    # Calling PlotLosses to plot the lossed in CNN Model
    #PlotLosses(cnnmodelfit)
    #plt.show()
    
    # ===================== 3.7. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ===================== 3.8. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ======================== 3.9. Plot the future ===============================
    
    # Plot the Test Predictions
    #PlotTestPrediction(df_test,df_predictions, ind)
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, ind)
    plt.show()
    
    # Printing the RMSE for Account
    rmseline = {'ACCOUNT_NO': ind, 'CNN_TESTRMSE': testrmse, 'CNN_FUTURERMSE': futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# Show all the plots    
plt.show()
# Creating a dataframe with RMSE against each Account
df_rmse = pd.DataFrame(rmselist)
print("CNN Model: Accounts with its RMSE")
print(df_rmse)

# Plot RMSE for each Cluster in barplot
df_rmse[['CNN_TESTRMSE','CNN_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Basic CNN model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Baisc CNN model for all accounts is {}".format(totaltime))


# Output:
#     ACCOUNT_NO  CNN_TESTRMSE  CNN_FUTURERMSE
#0       1196428      0.011394        0.012634
#1       1196711      0.398589        0.408600
#2  409000362497      0.032292        0.030062
#3  409000405747      0.123306        0.113044
#4  409000425051      0.000554        0.000554
#5  409000438611      0.051894        0.080983
#6  409000438620      0.001434        0.001434
#7  409000493201      0.001421        0.001421
#8  409000493210      0.001421        0.001421
#9  409000611074      0.000213        0.000213

# Total time to run the Baisc CNN model for all accounts is 4:19:57.254729


################### STEP 4: HYPERPARAMETER TUNING CNN MODEL WITH VARIOUS EPOCHS #####################

# ============= 4.1. Evaluating CNN Model for each ACCOUNT_NO ================

# Evaluating the CNN Model for each Account to tune the model and retreive best epoch value
for ind, eachact in actcount.items():

    # Getting TRANSACTION_AMOUNT attribute values for each ACCOUNT_NO into another dataframe
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M') #https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr
    
    # ================== 4.2. Split the data in train and test  ===================

    # Calling TrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    # Reshaping the Input Data as per Conv2D layers
    trainConv2D, testConv2D = ReshapeCNNData(trainingset, testset)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainConv2D.shape, testConv2D.shape))

    # Give a new line to clearly format the output
    Newline()
    
    # Output:
    # For Account 1196428 Training set is (34145, 1, 1) and Test set is (14634, 1, 1)
    
    # ========================== 4.3. Define models ===============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainConv2D, testConv2D, n_input)
    # Calling CreateCNNModel function to create the model object for CNN
    cnnmmodel = CreateCNNModel(trainConv2D, n_input, n_features)
    
    # ==================== 4.4. Build and Evaluate the Model ======================
    
    # Calling CompileFitTuneModel function to evaluate the deep learning model without any hyper-parameters
    dfepochscnn = CompileFitTuneModel(cnnmmodel, traingenerator, testgenerator, ind, trainConv2D)
    
    # Give a new line to clearly format the output
    Newline()
    
    # Finding out the best performed epoch by validating the minimum RMSE
    minrmse = dfepochscnn['RMSE'].min()
    df_best_epoch = dfepochscnn.loc[dfepochscnn['RMSE'] == minrmse, ['Epochs']]
    best_epochs.append(df_best_epoch['Epochs'].iloc[0])
   
# Retrieving the best epoch value
print("The best epochs value for each account are:")
print(best_epochs)
best_epoch = mode(best_epochs)
print("Epoch value that appeared most frequently is {}".format(best_epoch))

# Give a new line to clearly format the output
Newline()


# Output:
#The best epochs value for each account are:
#[50, 10, 100, 100, 50, 100, 50, 15, 25, 50]
#Epoch value that appeared most frequently is 50
   
"""  
# Hyperparameter tuning is not possible for CNN as it is throwing error 
# "ValueError: Received incompatible tensor with shape (1,) when attempting 
# to restore variable with shape (182,) and name layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE."

################# STEP 5: TUNE CNN MODEL WITH HYPERPARAMETERS #####################

# To tune the Deep Learning Model, kerastuner has been installed using "conda install -c conda-forge keras-tuner" [24]
# kerastuner was throwing error even after installing it with above command, so installed it with "pip3 install keras-tuner" [25]

#================== 5.1. Create a subset of data needed =======================

# Creating a subset dataset with only 'TRANSACTION_AMOUNT' attribute
df_transdata = dfbankdataset['TRANSACTION_AMOUNT']
df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
# There are many duplicate date entries in index, hence converting the index to monthly period
df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M') #https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr

# ================== 5.2. Split the data in train and test  ===================

# Calling TrainTestData function to split the data in train and test arrays
trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
# Reshaping the Input Data as per Conv2D layers
trainConv2D, testConv2D = ReshapeCNNData(trainingset, testset)
print("Training set is {} and Test set is {}".format(trainConv2D.shape, testConv2D.shape))

# Give a new line to clearly format the output
Newline()

# ================ 5.3. Tune model with RandomSearch ==========================

# To create an instance of RandomSearch class, RandomSearch function is called
# To maximize the accuracy "objective" is set to "loss"
# max_trails is set to 5 to limit the number of model variations to test
# executions_per_trial is set to 3 to limit the number of trials per variation

# Calling TuneRandomSearch function to tune and get the best model using RandomSearch
tuner = TuneRandomSearch(trainConv2D, testConv2D, CreateCNNModelHP, best_epoch)

# Summarize the results with best hyper-parameters
tuner.search_space_summary()

# Give a new line to clearly format the output
Newline()

# ================ 5.3. Retreive the best hyper parameters ====================

#Retreive the best performed hyper parameters
best_hp = tuner.get_best_hyperparameters()[0].values
best_units = best_hp['units']
best_learning_rate = best_hp['learning_rate']

# Print the best hyperparameter
print("The best hyperparameters are: {}".format(tuner.get_best_hyperparameters()[0].values))

# Give a new line to clearly format the output
Newline()

"""

####################### STEP 6: COMPLETE CNN MODEL  ##########################

start_time = datetime.now()
#print("Final CNN model start_time is - ", start_time)

# Hyperparameter tuning did not work for CNN Model. Hence considering best_units and best_Learning_rate got from LSTM model
best_units = 182
best_learning_rate = 0.01

# Cleat the list
rmselist.clear()

# ============= 6.1. Evaluating CNN Model for each ACCOUNT_NO ================

# Evaluate complete model for each account with best hyper parameters obtained from above steps
for ind, eachact in actcount.items():
    
    # Getting TRANSACTION_AMOUNT attribute values for each ACCOUNT_NO into another dataframe
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M') #https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr
    
    # ================== 6.2. Split the data in train and test  ===================

    # Calling TrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    # Reshaping the Input Data as per Conv2D layers
    trainConv2D, testConv2D = ReshapeCNNData(trainingset, testset)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainConv2D.shape, testConv2D.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 6.3. Create CNN Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainConv2D, testConv2D, n_input)
    # Calling FinalCNNModel function to create the model object for CNN
    cnnmodel = FinalCNNModel(trainConv2D, n_input, n_features, best_units)
    
    # ================ 6.4. Compile and Fit the CNN Model ========================
    
    # Calling FinalCompileFitModel function to fit the CNN model
    cnnmodelfit = FinalCompileFitModel(cnnmodel, traingenerator, testgenerator, best_epoch, best_learning_rate)
    
    # ===================== 6.5. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ===================== 6.6. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainConv2D, testConv2D, df_test, cnnmodelfit)
    
    # ======================== 6.7. Plot the future ===============================
    
    # Plot the Test Predictions
    #PlotTestPrediction(df_test,df_predictions, ind)
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, ind)
    plt.show()
    
    # Printing the RMSE for Account
    rmseline = {'ACCOUNT_NO': ind, 'CNN_TESTRMSE': testrmse, 'CNN_FUTURERMSE': futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()

# Show all the plots
plt.show()
# Printing the final RMSE against each account
df_finalrmse = pd.DataFrame(rmselist)
print("CNN Model: Accounts with its RMSE")
print(df_finalrmse)

# Plot RMSE for each Cluster in barplot
df_finalrmse[['CNN_TESTRMSE','CNN_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Final CNN model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final CNN model for all accounts is {}".format(totaltime))

# Output:
# CNN Model: Accounts with its RMSE
#     ACCOUNT_NO  CNN_TESTRMSE  CNN_FUTURERMSE
#0       1196428      0.019929        0.019929
#1       1196711      0.025568        0.025545
#2  409000362497      0.114113        0.114113
#3  409000405747      0.102083        0.097959
#4  409000425051      0.019880        0.019880
#5  409000438611      0.031178        0.031178
#6  409000438620      0.002511        0.002511
#7  409000493201      0.004798        0.004798
#8  409000493210      0.002025        0.002025
#9  409000611074      0.007254        0.007254

# Total time to run the Final CNN model for all accounts is 2:35:35.374095