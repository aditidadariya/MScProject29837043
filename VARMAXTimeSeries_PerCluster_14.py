#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 01:23:19 2022

Student ID: 29837043
File: VARMAXTimeSeries_PerCluster_14.py

This file contains multivariate time series forecasting in VARMAX model for each cluster formed.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

from sklearn.metrics import mean_squared_error
import math 
from statistics import mean
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
warnings.filterwarnings('ignore')


################### STEP 2: READ THE CLEANED DATA FILE ########################

# Calling ReadDataset to read the clean dataset obtained from DataPreProcessing_DataVisualization
dfbankdataset = ReadDataset(location_of_file, cleanfile)
# Calling ReadDataset funciton to read the data of all clusters
dfcluster0 = ReadDataset(location_of_file, cluster0data)
dfcluster1 = ReadDataset(location_of_file, cluster1data)
dfcluster2 = ReadDataset(location_of_file, cluster2data)
dfcluster3 = ReadDataset(location_of_file, cluster3data)
dfcluster4 = ReadDataset(location_of_file, cluster4data)

# Save all the dataframes into a list
ClusterData = [dfcluster0, dfcluster1, dfcluster2, dfcluster3, dfcluster4]

###################### STEP 3: STATIONARITY TEST ##############################
  
# Calling ADFStationarityTest function to test the data for stationarity using dickey-fuller-test
ADFStationarityTest(dfbankdataset['TRANSACTION_AMOUNT'])

# Give a new line to clearly format the output
Newline()

# Calling ADFStationarityTest function to test the data for stationarity using dickey-fuller-test
ADFStationarityTest(dfbankdataset['TRANSACTION_DETAILS'])

# Give a new line to clearly format the output
Newline()

# Output:
#Augmented Dickey-Fuller Test Results:
#ADF Test Statistics               -78.587497
#p-value                             0.000000
##Lag Used                          16.000000
#Number of Observations Used    116184.000000
#Critical Value (1%)                -3.430406
#Critical Value (5%)                -2.861565
#Critical Value (10%)               -2.566783
#dtype: float64
#The time series data has not unit roots and hence it is stationary


#Augmented Dickey-Fuller Test Results:
#ADF Test Statistics               -26.197807
#p-value                             0.000000
##Lag Used                          71.000000
#Number of Observations Used    116129.000000
#Critical Value (1%)                -3.430406
#Critical Value (5%)                -2.861565
#Critical Value (10%)               -2.566783
#dtype: float64
#The time series data has not unit roots and hence it is stationary

################ STEP 4: DETERMINE ROLLING STATISTICS #########################
    
# Calling RollingStats function to display the Rolling Mean and Rolling Std
RollingStats(dfbankdataset['TRANSACTION_AMOUNT'])
RollingStats(dfbankdataset['TRANSACTION_DETAILS'])

# Give a new line to clearly format the output
Newline()

# As per the rolling plots Mean and standard deviation for TRANSACTION AMOUNT is more in the beginning however it floats
# towards 0 eventually, which means the mean becomes stable towards the end. Whereas
# the mean and standard deviation for TRANSACTION DETAILS seems almost constant throughout showing that it was stable all the time


############# STEP 5: DETERMINE DEPENDENCY OF ATTRIBUTES ######################

print('TRANSACTION_AMOUNT causes TRANSACTION_DETAILS?\n')
granger_1 = grangercausalitytests(dfbankdataset[['TRANSACTION_DETAILS', 'TRANSACTION_AMOUNT']], 4)

# Give a new line to clearly format the output
Newline()

print('\TRANSACTION_DETAILS causes TRANSACTION_AMOUNT?\n')
granger_2 = grangercausalitytests(dfbankdataset[['TRANSACTION_AMOUNT', 'TRANSACTION_DETAILS']], 4)

# Give a new line to clearly format the output
Newline()

# p value is 0 is all the lags, hence its clear that there is no dependency between Transaction details and Transaction amount

# Output:
    
#TRANSACTION_AMOUNT causes TRANSACTION_DETAILS?

#Granger Causality
#number of lags (no zero) 1
#ssr based F test:         F=47.7199 , p=0.0000  , df_denom=116197, df_num=1
#ssr based chi2 test:   chi2=47.7211 , p=0.0000  , df=1
#likelihood ratio test: chi2=47.7113 , p=0.0000  , df=1
#parameter F test:         F=47.7199 , p=0.0000  , df_denom=116197, df_num=1

#Granger Causality
#number of lags (no zero) 2
#ssr based F test:         F=49.7570 , p=0.0000  , df_denom=116194, df_num=2
#ssr based chi2 test:   chi2=99.5182 , p=0.0000  , df=2
#likelihood ratio test: chi2=99.4756 , p=0.0000  , df=2
#parameter F test:         F=49.7570 , p=0.0000  , df_denom=116194, df_num=2

#Granger Causality
#number of lags (no zero) 3
#ssr based F test:         F=31.8132 , p=0.0000  , df_denom=116191, df_num=3
#ssr based chi2 test:   chi2=95.4453 , p=0.0000  , df=3
#likelihood ratio test: chi2=95.4061 , p=0.0000  , df=3
#parameter F test:         F=31.8132 , p=0.0000  , df_denom=116191, df_num=3

#Granger Causality
#number of lags (no zero) 4
#ssr based F test:         F=21.2019 , p=0.0000  , df_denom=116188, df_num=4
#ssr based chi2 test:   chi2=84.8141 , p=0.0000  , df=4
#likelihood ratio test: chi2=84.7832 , p=0.0000  , df=4
#parameter F test:         F=21.2019 , p=0.0000  , df_denom=116188, df_num=4


#\TRANSACTION_DETAILS causes TRANSACTION_AMOUNT?


#Granger Causality
#number of lags (no zero) 1
#ssr based F test:         F=23.7042 , p=0.0000  , df_denom=116197, df_num=1
#ssr based chi2 test:   chi2=23.7048 , p=0.0000  , df=1
#likelihood ratio test: chi2=23.7024 , p=0.0000  , df=1
#parameter F test:         F=23.7042 , p=0.0000  , df_denom=116197, df_num=1

#Granger Causality
#number of lags (no zero) 2
#ssr based F test:         F=20.5731 , p=0.0000  , df_denom=116194, df_num=2
#ssr based chi2 test:   chi2=41.1480 , p=0.0000  , df=2
#likelihood ratio test: chi2=41.1407 , p=0.0000  , df=2
#parameter F test:         F=20.5731 , p=0.0000  , df_denom=116194, df_num=2

#Granger Causality
#number of lags (no zero) 3
#ssr based F test:         F=15.0432 , p=0.0000  , df_denom=116191, df_num=3
#ssr based chi2 test:   chi2=45.1323 , p=0.0000  , df=3
#likelihood ratio test: chi2=45.1235 , p=0.0000  , df=3
#parameter F test:         F=15.0432 , p=0.0000  , df_denom=116191, df_num=3

#Granger Causality
#number of lags (no zero) 4
#ssr based F test:         F=11.9609 , p=0.0000  , df_denom=116188, df_num=4
#ssr based chi2 test:   chi2=47.8472 , p=0.0000  , df=4
#likelihood ratio test: chi2=47.8373 , p=0.0000  , df=4
#parameter F test:         F=11.9609 , p=0.0000  , df_denom=116188, df_num=4

################### STEP 6: BUILD VARMAX MODEL PER CLUSTER #######################

# https://github.com/nachi-hebbar/Multivariate-Time-Series-Forecasting/blob/main/VAR_Model%20(1).ipynb
# https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/

# ================== 6.1. Group the dataset with the Accounts  ================

start_time = datetime.now()
#print("Basic ARIMA model start_time is - ", start_time)

# Clear list
rmselist.clear()

# ============= 6.2. Evaluating VARMAX Model for each Cluster ================

# Evaluate VARMAX model for each cluster with the best hyperparameters found previuously
# Creating a forecast and getting the Root Mean Squared Error
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]] = dfeach[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
    
    # ================== 6.3. Split the data in train and test  ===============
    
    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ====================== 6.4. Evaluate VARMAX Model ========================
    
    # Calling EvaluateVARMAXTestModel and EvaluateVARMAXFutureModel to evaluate the VARMAX model with p =4 and q =0
    td_testrmse, ta_testrmse, df_testpredictions = EvaluateVARMAXTestModel(df_train, df_test, 4, 0)
    td_futurermse, ta_futurermse, df_futurepredictions = EvaluateVARMAXFutureModel(df_train, df_test, 4, 0)
    
    # ======================== 6.5. Plot Predictions ==========================
    
    # Plot the VARMAX Predictions
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,"Cluster"+str(index))
    
    # =========================== 6.6. Store RMSE =============================
    
    rmseline = {"CLUSTER": "Cluster"+str(index), "TD_TESTRMSE": td_testrmse, "TD_FUTURERMSE": td_futurermse, "TA_TESTRMSE": ta_testrmse, "TA_FUTURERMSE": ta_futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# ==================== 6.7. Plot RMSE of all clusters =========================

# Create dataframe containing Cluster with its RMSE 
df_rmse = pd.DataFrame(rmselist)
print("VARMAX Model: Clusters with its RMSE")
print(df_rmse)

# Plot RMSE for each Cluster in barplot
df_rmse[['TD_TESTRMSE','TD_FUTURERMSE','TA_TESTRMSE','TA_FUTURERMSE']].plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Basic VARMAX model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the basic VARMAX model for all clusters is {}".format(totaltime))

# Give a new line to clearly format the output
Newline()

#Output:
#VARMAX Model: Clusters with its RMSE
#    CLUSTER  TD_TESTRMSE  TD_FUTURERMSE  TA_TESTRMSE  TA_FUTURERMSE
#0  Cluster0    31.895246      78.945231     0.058416       0.005461
#1  Cluster1    33.759787      47.441194     0.065123       0.029650
#2  Cluster2    28.801888      61.180945     0.253663       0.283935
#3  Cluster3     9.279311      12.598664     0.005590       0.002286
#4  Cluster4     9.626456      39.777013     0.031149       0.008779

# Total time to run the basic VARMAX model for all clusters is 0:15:18.291976

############# STEP 7: TUNE VARMAX MODEL WITH VARIOUS p,d,q VALUES ##############

# - d will remain as 0 because the dataset is stationary and do no need any differences
# - Reference: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# Evaluate the VARMAX model with various p_values and q_values provided in Config_1.py file
# p_values are [1, 2, 4, 6] and q_values is range(0, 4)

# =========== 7.1. Evaluating VARMAX model to get best order values ============

# df_result dataframe is defined to store the results at the end
df_result = pd.DataFrame()

# Looping the evalutions of VARMAX model for each cluster
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_clusterdata = pd.DataFrame()
    df_clusterdata[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]] = dfeach[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_clusterdata.index = pd.DatetimeIndex(df_clusterdata.index).to_period('M')
     
    # ================ 7.2. Split the data in train and test  =================

    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_clusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 7.3. TUNE ARIMA Model =========================
    
    # Calling EvaluateTuneVARMAXModels function to evaluate VARMAX Model with various p,d,q values
    df_bestvarmax = EvaluateTuneVARMAXModels(df_train, df_test, "Cluster"+str(index), p_values, q_values)
    # Storing the results
    df_result = pd.concat([df_result,df_bestvarmax], keys=["CLUSTER", "ORDER"], ignore_index=True)
    
    # Give a new line to clearly format the output
    Newline()
    
# ==================== 7.4. Print RMSE of all clusters =========================

print("Best order with its RMSE for all clusters:")
print(df_result)

# Give a new line to clearly format the output
Newline()

# Output:
#Best order with its RMSE for all clusters:
#    CLUSTER   ORDER
#0  Cluster0  [6, 0]
#1  Cluster1  [2, 2]
#2  Cluster2  [6, 0]
#3  Cluster3  [2, 0]
#4  Cluster4  [6, 1]

######################## STEP 8: FINAL VARMAX MODEL ############################

start_time = datetime.now()
#print("Final VARMAX model start_time is - ", start_time)

rmselist.clear()

# Evaluating VARMAX Model with the best order of p,d,q found after tuning the model

# =========== 8.1. Evaluating VARMAX model to get best order values ============

# Looping the evalutions of VARMAX model for each cluster
for index, dfeach in enumerate(ClusterData):
    
    # Storing the subset of data into new dataframe
    df_finalclusterdata = pd.DataFrame()
    df_finalclusterdata[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]] = dfeach[["TRANSACTION_DETAILS","TRANSACTION_AMOUNT"]]
    # There are many duplicate date entries in index, hence converting the index to monthly period
    df_finalclusterdata.index = pd.DatetimeIndex(df_finalclusterdata.index).to_period('M')
    
    # =============== 8.2. Split the data in train and test  ==================
    
    # Calling SplitTrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_finalclusterdata)
    print("For {} Training set is {} and Test set is {}".format("Cluster"+str(index), trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================= 8.3. Retreive p,d,q values ======================
    
    # Get the p, d, q values out of the ORDER column in dataframe
    order = df_result.loc[df_result['CLUSTER']=="Cluster"+str(index) , ['ORDER']]
    ordervalues = order.iloc[0]
    p = ordervalues.iloc[0][0]
    q = ordervalues.iloc[0][1]

    # ====================== 8.4. Evaluate VARMAX Model ========================
    
    td_testrmse, ta_testrmse, df_testpredictions = EvaluateVARMAXTestModel(df_train, df_test, p, q)
    td_futurermse, ta_futurermse, df_futurepredictions = EvaluateVARMAXFutureModel(df_train, df_test, p, q)
    
    # ======================== 8.5. Plot Predictions ==========================
    
    # Plot the VARMAX Predictions
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,"Cluster"+str(index))
    
    # =========================== 8.6. Store RMSE =============================
    
    rmseline = {"CLUSTER": "Cluster"+str(index), "TD_TESTRMSE": td_testrmse, "TD_FUTURERMSE": td_futurermse, "TA_TESTRMSE": ta_testrmse, "TA_FUTURERMSE": ta_futurermse}
    # Updating the outputlist with the line list
    rmselist.append(rmseline)
    
    # Give a new line to clearly format the output
    Newline()
    
# ==================== 8.7. Plot RMSE of all clusters =========================

# Create dataframe containing ACCOUNT_NO with its RMSE 
df_finalrmse = pd.DataFrame(rmselist)
print("VARMAX Model: Clusters with its RMSE")
print(df_finalrmse)
# Plot RMSE for each ACCOUNT_NO in barplot
df_finalrmse.plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Final VARMAX model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Final VARMAX model for all clusters is {}".format(totaltime))

#VARMAX Model: Clusters with its RMSE
#    CLUSTER  TD_TESTRMSE  TD_FUTURERMSE  TA_TESTRMSE  TA_FUTURERMSE
#0  Cluster0    31.893163      79.074179     0.058418       0.005917
#1  Cluster1    33.047624      47.506673     0.065052       0.029624
#2  Cluster2    28.801374      61.224498     0.253663       0.290735
#3  Cluster3     9.278088      14.378374     0.005590       0.002485
#4  Cluster4     9.604047      39.065566     0.031036       0.011705

#Total time to run the Final VARMAX model for all clusters is 0:13:50.581230