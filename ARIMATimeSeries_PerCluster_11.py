#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:40:29 2022

Student ID: 29837043
File: ARIMATimeSeries_PerCluster_11.py

This file contains univariate time series forecasting in ARIMA model for each cluster formed.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
#from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import time
from datetime import datetime
import time
# Import Config.py file
from Config_1 import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess_2 import *
# Import FuncLibVisual.py file
from FuncLibVisual_3 import *
# Import FuncLibModel.py file
from FuncLibModel_4 import *

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

################### STEP 3: BUILD THE BASIC ARIMA MODEL #######################

# ================== 3.1. Group the dataset with the Accounts  ================

start_time = datetime.now()
#print("Basic ARIMA model start_time is - ", start_time)

# Cleat the list
rmselist.clear()

# ========== 3.2. Evaluating the basic ARIMA Model for each Cluster  ==========

# Looping the evalutions of ARIMA model for each cluster
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata["TRANSACTION_AMOUNT"] = dfeach["TRANSACTION_AMOUNT"]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
    
    # ================== 3.3. Split the data in train and test  ===============
    
    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # Output:
    # For Cluster0 Training set is (41520, 1) and Test set is (17795, 1)

    # ====================== 3.4. Evaluate ARIMA Model ========================
        
    # Calling EvaluateARIMAModel function to train and predict basic ARIMA model with p,d,q as 3,0,4
    testrmse, df_testpredictions = EvaluateARIMATestModel(df_train, df_test, 3, 0, 4, "Cluster"+str(index))
    futurermse, df_futurepredictions = EvaluateARIMAFutureModel(df_train, df_test, 3, 0, 4, "Cluster"+str(index))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================== 3.5. Plot Predictions ==========================
    
    # Plot the ARIMA Predictions
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,"Cluster"+str(index))
    
    # =========================== 3.6. Store RMSE =============================
    
    # Store Root Mean Squared Error for each Account
    rmseline = {'CLUSTER': "Cluster"+str(index), 'TESTRMSE': testrmse, 'FUTURERMSE': futurermse}
    rmselist.append(rmseline)

# ==================== 3.7. Plot RMSE of all accounts =========================

# Create dataframe containing ACCOUNT_NO with its RMSE 
df_rmse = pd.DataFrame(rmselist)
print("ARIMA Model: Clusters with its RMSE")
print(df_rmse)
# Plot RMSE for each ACCOUNT_NO in barplot
df_rmse.plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Basic ARIMA model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the basic ARIMA model for all clusters is {}".format(totaltime))

# Give a new line to clearly format the output
Newline()

# Output:
#ARIMA Model: Clusters with its RMSE
#    CLUSTER  TESTRMSE  FUTURERMSE
#0  Cluster0  0.400080    0.400080
#1  Cluster1  0.046997    0.046997
#2  Cluster2  0.285307    0.285307
#3  Cluster3  0.007626    0.007626
#4  Cluster4  0.032458    0.032458

# Total time to run the basic ARIMA model for all clusters is 0:05:05.379784

############# STEP 4: TUNE ARIMA MODEL WITH VARIOUS p,d,q VALUES ##############

# - d will remain as 0 because the dataset is stationary and do no need any differences
# - Reference: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# =========== 4.1. Evaluating ARIMA model to get best order values ============

# df_result dataframe is defined to store the results at the end
df_result = pd.DataFrame()

# Looping the evalutions of ARIMA model for each account
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata["TRANSACTION_AMOUNT"] = dfeach["TRANSACTION_AMOUNT"]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
     
    # ================ 4.2. Split the data in train and test  =================

    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 4.3. TUNE ARIMA Model =========================
    
    # Calling EvaluateTuneARIMAmodels function to evaluate ARIMA Model with various p,d,q values
    df_bestarima = EvaluateTuneARIMAModels(df_train, df_test, "Cluster"+str(index), p_values, d_values, q_values)
    # Storing the results
    df_result = pd.concat([df_result,df_bestarima], keys=["ACCOUNT_NO", "ORDER", "RMSE"], ignore_index=True)
    
    # Give a new line to clearly format the output
    Newline()
    
# ==================== 4.4. Plot RMSE of all accounts =========================

df_result.columns = ["CLUSTER", "ORDER", "RMSE"]
print("Best order with its RMSE for all clusters:")
print(df_result)
df_result.plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

#Output:
#Best order with its RMSE for all clusters:
#  ACCOUNT_NO      ORDER      RMSE
#0   Cluster0  [2, 0, 5]  0.399855
#1   Cluster1  [3, 0, 5]  0.046986
#2   Cluster2  [5, 0, 4]  0.285252
#3   Cluster3  [5, 0, 5]  0.007622
#4   Cluster4  [3, 0, 5]  0.032077

######################## STEP 5: FINAL ARIMA MODEL ############################

start_time = datetime.now()
#print("Final ARIMA model start_time is - ", start_time)

# Clear the list
rmselist.clear()

# =========== 5.1. Evaluating ARIMA model to get best order values ============

# Evaluating ARIMA Model with the best order of p,d,q found after tuning the model

# Looping the evalutions of ARIMA model for each account
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata["TRANSACTION_AMOUNT"] = dfeach["TRANSACTION_AMOUNT"]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
    
    # =============== 5.2. Split the data in train and test  ==================
    
    # Calling SplitTrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================= 5.3. Retreive p,d,q values ======================
    
    # Get the p, d, q values out of the ORDER column in dataframe
    order = df_result.loc[df_result['CLUSTER']=="Cluster"+str(index) , ['ORDER']]
    ordervalues = order.iloc[0]
    p = ordervalues.iloc[0][0]
    d = ordervalues.iloc[0][1]
    q = ordervalues.iloc[0][2]
    
    # ====================== 5.4. Evaluate ARIMA Model ========================
        
    # Calling EvaluateARIMAModel function to train and predict basic ARIMA model with p,d,q
    testrmse, df_testpredictions = EvaluateARIMATestModel(df_train, df_test, p, d, q, "Cluster"+str(index))
    futurermse, df_futurepredictions = EvaluateARIMAFutureModel(df_train, df_test, p, d, q, "Cluster"+str(index))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================== 5.5. Plot Predictions ==========================
    
    # Seting the plot size
    plt.rcParams["figure.figsize"] = (8,4)
    # Plot the ARIMA Predictions
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,"Cluster"+str(index))
    
    # =========================== 5.6. Store RMSE =============================
    
    # Store Root Mean Squared Error for each Account
    rmseline = {'CLUSTER': "Cluster"+str(index), 'TESTRMSE': testrmse, 'FUTURERMSE': futurermse}
    #rmseline = {'CLUSTER': "Cluster"+str(index), 'RMSE': rmse}
    rmselist.append(rmseline)

# ==================== 5.7. Plot RMSE of all accounts =========================

# Create dataframe containing ACCOUNT_NO with its RMSE 
df_finalrmse = pd.DataFrame(rmselist)
print("ARIMA Model: Clusters with its RMSE")
print(df_finalrmse)
# Plot RMSE for each ACCOUNT_NO in barplot
df_finalrmse.plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Final ARIMA model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final ARIMA model for all clusters is {}".format(totaltime))

#ARIMA Model: Clusters with its RMSE
#    CLUSTER  TESTRMSE  FUTURERMSE
#0  Cluster0  0.399855    0.399855
#1  Cluster1  0.046986    0.046986
#2  Cluster2  0.285252    0.285252
#3  Cluster3  0.007622    0.007622
#4  Cluster4  0.032077    0.032077

#Total time to run the Final ARIMA model for all clusters is 0:07:43.708676


