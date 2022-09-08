#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:32:08 2022

Student ID: 29837043
File: LSTMTimeSeries_PerAccount_8.py

This file contains univariate time series forecasting in LSTM model for each account.

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

from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore")

################### STEP 2: READ THE CLEANED DATA FILE ########################

# Calling ReadDataset to read the clean dataset obtained from DataPreProcessing_DataVisualization
dfbankdataset = ReadDataset(location_of_file, cleanfile)

###################### STEP 3: BUILD BASIC LSTM MODEL  ########################

start_time = datetime.now()
#print("Basic LSTM model start_time is - ", start_time)

# =============== 3.1. Group the dataset for based ACCOUNT_NO =================

# Grouping the ACCOUNT_NO based on maximum number of records for each account in the dfbankdataset dataframe 
actcount = dfbankdataset.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()

# Clear the list 
rmselist.clear()

# ============= 3.2. Evaluating LSTM Model for each ACCOUNT_NO ================

# Building the basic model for each account, creating a forecast and getting the Root Mean Squared Error
for ind, eachact in actcount.items():
    
    # Getting TRANSACTION_AMOUNT attribute values for each ACCOUNT_NO into another dataframe
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO'] == ind ,['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M') #https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr
    
    # ================== 3.3. Split the data in train and test  ===================
    
    # Calling TrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # Output:
    # For Account 1196428 Training set is (34145, 1) and Test set is (14634, 1)
    
    # ========================= 3.4. Create LSTM Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainingset, testset, n_input)
    # Calling CreateLSTMModel function to create the model object for LSTM
    lstmmodel = CreateLSTMModel(trainingset, n_input, n_features)

    #Output:
    #Model: "sequential_1"
    #_________________________________________________________________
    # Layer (type)                Output Shape              Param #   
    #=================================================================
    # lstm_4 (LSTM)               (None, 10, 64)            16896     
                                                                     
    # dropout_3 (Dropout)         (None, 10, 64)            0         
                                                                     
    # lstm_5 (LSTM)               (None, 32)                12416     
                                                                     
    # dropout_4 (Dropout)         (None, 32)                0         
                                                                     
    # dense_2 (Dense)             (None, 1)                 33        
                                                                     
    #=================================================================
    #Total params: 29,345
    #Trainable params: 29,345
    #Non-trainable params: 0
    
    
    # ================ 3.5. Compile and Fit the LSTM Model ========================
    
    # Calling CompileFitModel function to fit the LSTM model
    lstmmodelfit = CompileFitModel(lstmmodel, traingenerator, testgenerator)
    
    # ===================== 3.6. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ===================== 3.7. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ======================== 3.8. Plot the future ===============================
    
    # Plot the Test Predictions
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, ind)
    
    # Printing the RMSE for Account
    #print("The Root Mean Squared Error is: {}".format(rmse))
    rmseline = {'ACCOUNT_NO': ind, 'LSTM_TESTRMSE': testrmse, 'LSTM_FUTURERMSE': futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# Show all the plots    
plt.show()
# Creating a dataframe with RMSE against each Account
df_rmse = pd.DataFrame(rmselist)
print("LSTM Model: Accounts with its RMSE")
print(df_rmse)

# Plot RMSE for each Cluster in barplot
df_rmse[['LSTM_TESTRMSE','LSTM_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Basic LSTM model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Basic LSTM model for all accounts is {}".format(totaltime))

# Output:
##LSTM Model: Accounts with its RMSE
#     ACCOUNT_NO  LSTM_TESTRMSE  LSTM_FUTURERMSE
#0       1196428       0.019859         0.015418
#1       1196711       0.106888         0.114390
#2  409000362497       0.006949         0.009964
#3  409000405747       0.088033         0.073018
#4  409000425051       0.004870         0.004870
#5  409000438611       0.056622         0.058319
#6  409000438620       0.001924         0.001924
#7  409000493201       0.000738         0.000738
#8  409000493210       0.000832         0.000832
#9  409000611074       0.002929         0.002929

# Total time to run the Basic LSTM model for all accounts is 4:33:24.962375

################### STEP 4: HYPERPARAMETER TUNING LSTM MODEL WITH VARIOUS EPOCHS #####################

# https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

# ============= 4.1. Evaluating LSTM Model for each ACCOUNT_NO ================

# Evaluating the LSTM Model for each Account to tune the model and retreive best epoch value
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
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))

    # Give a new line to clearly format the output
    Newline()
    
    # Output:
    # For Account 1196428 Training set is (34145, 1) and Test set is (14634, 1)
    
    # ========================== 4.3. Define models ===============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainingset, testset, n_input)
    # Calling CreateLSTMModel function to create the model object for LSTM
    lstmmodel = CreateLSTMModel(trainingset, n_input, n_features)
    
    # ==================== 4.4. Build and Evaluate the Model ======================
    
    # Calling CompileFitTuneModel function to evaluate the deep learning model without any hyper-parameters
    dfepochslstm = CompileFitTuneModel(lstmmodel, traingenerator, testgenerator, ind, trainingset)
    
    # Give a new line to clearly format the output
    Newline()
    
    # Finding out the best performed epoch by validating the minimum RMSE
    minrmse = dfepochslstm['RMSE'].min()
    df_best_epoch = dfepochslstm.loc[dfepochslstm['RMSE'] == minrmse, ['Epochs']]
    best_epochs.append(df_best_epoch['Epochs'].iloc[0])
   
# Retrieving the best epoch value
print("The best epochs value for each account are:")
print(best_epochs)
best_epoch = mode(best_epochs)
print("Epoch value that appeared most frequently is {}".format(best_epoch))

# Give a new line to clearly format the output
Newline()

# Output:
# The best epochs value for each account are:
#[20, 50, 25, 100, 15, 50, 20, 20, 20, 100]
#Epoch value that appeared most frequently is 20

################# STEP 5: TUNE LSTM MODEL WITH HYPERPARAMETERS #####################

# https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/

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
print("Training set is {} and Test set is {}".format(trainingset.shape, testset.shape))

# Give a new line to clearly format the output
Newline()

# ================ 5.3. Tune model with RandomSearch ==========================

# To create an instance of RandomSearch class, RandomSearch function is called
# To maximize the perpormance "objective" is set to "loss"
# max_trails is set to 5 to limit the number of model variations to test
# executions_per_trial is set to 3 to limit the number of trials per variation

# Calling TuneRandomSearch function to tune and get the best model using RandomSearch
tuner = TuneRandomSearch(trainingset, testset, CreateLSTMModelHP, best_epoch)
#tuner = TuneLSTMRandomSearch(traingenerator, testgenerator, best_epoch)

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

# Output:
# Search space summary
#Default search space size: 2
#units (Int)
#{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 300, 'step': 50, 'sampling': None}
#learning_rate (Choice)
#{'default': 0.01, 'conditions': [], 'values': [0.01, 0.0001], 'ordered': True}

#The best hyperparameters are: {'units': 182, 'learning_rate': 0.01}


####################### STEP 6: FINAL LSTM MODEL  ##########################

start_time = datetime.now()
#print("Final LSTM model start_time is - ", start_time)

# Cleat the list
rmselist.clear()

# ============= 6.1. Evaluating LSTM Model for each ACCOUNT_NO ================

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
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 6.3. Create LSTM Model ============================
    
    # Calling ProduceGenerator to get the train and test generators
    traingenerator, testgenerator = ProduceGenerator(trainingset, testset, n_input)
    # Calling FinalLSTMModel function to create the model object for LSTM
    lstmmodel = FinalLSTMModel(trainingset, n_input, n_features, best_units)
    
    # ================ 6.4. Compile and Fit the LSTM Model ========================
    
    # Calling FinalCompileFitModel function to fit the LSTM model
    lstmmodelfit = FinalCompileFitModel(lstmmodel, traingenerator, testgenerator, best_epoch, best_learning_rate)
    
    # ===================== 6.5. Predict the test data ============================
    
    # Calling TestPrediction function to create prediction on test data
    testrmse, df_testpredictions = TestPrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ===================== 6.6. Predict the future ===============================

    # Calling FuturePrediction function to create future predictios using testset
    futurermse, df_futurepredictions = FuturePrediction(trainingset, testset, df_test, lstmmodelfit)
    
    # ======================== 6.7. Plot the future ===============================
    
    # Plot the Test Predictions
    PlotTestFuturePrediction(df_test, df_testpredictions, df_futurepredictions, ind)
    
    # Printing the RMSE for Account
    rmseline = {'ACCOUNT_NO': ind, 'LSTM_TESTRMSE': testrmse, 'LSTM_FUTURERMSE': futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    

# Show all the plots
plt.show()
# Printing the final RMSE against each account
df_finalrmse = pd.DataFrame(rmselist)
print("LSTM Model: Accounts with its RMSE")
print(df_finalrmse)

# Plot RMSE for each Cluster in barplot
df_finalrmse[['LSTM_TESTRMSE','LSTM_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

end_time = datetime.now()
#print("Final LSTM model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final LSTM model for all accounts is {}".format(totaltime))


#Output:
# LSTM Model: Accounts with its RMSE
#     ACCOUNT_NO  LSTM_TESTRMSE  LSTM_FUTURERMSE
#0       1196428       0.020162         0.020162
#1       1196711       0.008897         0.008897
#2  409000362497       0.145800         0.134205
#3  409000405747       0.159691         0.118021
#4  409000425051       0.003227         0.003227
#5  409000438611       0.046779         0.046779
#6  409000438620       0.001353         0.001353
#7  409000493201       0.011570         0.011570
#8  409000493210       0.001284         0.001284
#9  409000611074       0.002359         0.002359

#Total time to run the Final LSTM model for all accounts is 3:30:22.896829