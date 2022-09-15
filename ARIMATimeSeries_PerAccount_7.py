#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:40:29 2022

Student ID: 29837043
File: ARIMATimeSeries_PerAccount_7.py

This file contains univariate time series forecasting in ARIMA model for each account.

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
dfbankdataset = ReadDataset(location_of_file, cleanfile)

###################### STEP 3: STATIONARITY TEST ##############################
  
# Calling ADFStationarityTest function to test the data for stationarity using Augmented Dickey-Fuller test
ADFStationarityTest(dfbankdataset['TRANSACTION_AMOUNT'])     

# Give a new line to clearly format the output
Newline()

# OUTPUT:
#Augmented Dickey-Fuller Test Results:
#ADF Test Statistics               -78.587497
#p-value                             0.000000
##Lag Used                          16.000000
#Number of Observations Used    116184.000000
#Critical Value (1%)                -3.430406
#Critical Value (5%)                -2.861565
#Critical Value (10%)               -2.566783
#dtype: float64

#The time series data has no unit roots and hence it is stationary

################ STEP 4: DETERMINE ROLLING STATISTICS #########################
    
# Calling RollingStats function to display the Rolling Mean and Rolling Std
RollingStats(dfbankdataset['TRANSACTION_AMOUNT'])

# The plot of rolling mean and standard deviation shows that the mean was fluctuation in the beginning 
# however, eventually it became stable towards end.

##################### STEP 5: PLOT ACF AND PACF ###############################

# Plot Auto-Correlation and Partial Auto-Correlation plot

#================== 5.1. Create a subset of data needed =======================

# Creating a subset dataset with only 'TRANSACTION_AMOUNT' attribute
df_transdata = dfbankdataset['TRANSACTION_AMOUNT']
df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])

#======================= 5.2. Plot ACF and PACF ===============================

# Calling PlotCorrData function to plot the auto-correction plots between original data and correlated data
PlotCorrData(df_transdata)

# Wait until the graph is displayed
time.sleep(20)

# Give a new line to clearly format the output
Newline()

# Observation:
# - Looking at the Autocorrelation and Partial autocorrelation graphs above it is clear that there is no exponential growth or trend, 
# - inspite of having sudden drop from 0 to 1-4 and then the graph floats around 0
# - Hence the autocorrelation/moving average can be taken as 0 or 4. Considering MA = q = 4
# - and autoregressive p can be 3 as the graph suddenly drops to 1 and then spike slightly to 3 downwards. 
# - However, the upper bound and lower bound is very close to 0 so p value can be taken as 2 or 3
# - also the data is stationary, so there was no differening/shift done, which makes d to be 0. 
# - ARMA model is not considered even when d is 0 because ARMA model is deprecated and is throwing error. [1]
# - This univariate dataset is not seasonal, hence SARIMA model is not suitable for this time series dataset
# - Implementing ARIMA model for this dataset, where p = 3, d = 0 and q = 4

################### STEP 6: BUILD THE BASIC ARIMA MODEL #######################

# ================== 6.1. Group the dataset with the Accounts  ================

start_time = datetime.now()
#print("Basic ARIMA model start_time is - ", start_time)

# Clear the list
rmselist.clear()

# Getting the number of rows for each account 
actcount = dfbankdataset.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()

# ========== 6.2. Evaluating the basic ARIMA Model for each Account  ==========

# Looping the evalutions of ARIMA model for each account
for ind, eachact in actcount.items():

    # Getting all the rows for each account from the dataset
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period 
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M')    # [2]
    
    # ================== 6.3. Split the data in train and test  ===============
    
    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # Output:
    # For Account 1196428 Training set is (34145, 1) and Test set is (14634, 1)
    
    # ====================== 6.4. Evaluate ARIMA Model ========================
        
    # Calling EvaluateARIMAModel function to train and predict basic ARIMA model with p,d,q as 3,0,4
    testrmse, df_testpredictions = EvaluateARIMATestModel(df_train, df_test, 3, 0, 4, ind)
    futurermse, df_futurepredictions = EvaluateARIMAFutureModel(df_train, df_test, 3, 0, 4, ind)
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================== 6.5. Plot Predictions ==========================
    
    # Setting the plot size
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,ind)
    
    # =========================== 6.6. Store RMSE =============================
    
    # Store Root Mean Squared Error for each Account
    rmseline = {'ACCOUNT_NO': str(ind), 'TESTRMSE': testrmse, 'FUTURERMSE': futurermse}
    rmselist.append(rmseline)

# ==================== 6.7. Plot RMSE of all accounts =========================

# Create dataframe containing ACCOUNT_NO with its RMSE 
df_rmse = pd.DataFrame(rmselist)
print("ARIMA Model: Accounts with its RMSE")
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
print("Total time to run the basic ARIMA model for all accounts is {}".format(totaltime))

# Give a new line to clearly format the output
Newline()

# Output:
#ARIMA Model: Accounts with its RMSE
#     ACCOUNT_NO  TESTRMSE  FUTURERMSE
#0       1196428  0.025124    0.025124
#1       1196711  0.876216    0.876216
#2  409000362497  0.285307    0.285307
#3  409000405747  0.151571    0.151571
#4  409000425051  0.011575    0.011575
#5  409000438611  0.096860    0.096860
#6  409000438620  0.000231    0.000231
#7  409000493201  0.016103    0.016103
#8  409000493210  0.000328    0.000328
#9  409000611074  0.032458    0.032458

# Total time to run the basic ARIMA model for all accounts is 0:04:53.490331

############# STEP 7: TUNE ARIMA MODEL WITH VARIOUS p,d,q VALUES ##############

# - d will remain as 0 because the dataset is stationary and do no need any differences
# values of p,d,and q are taken from Config_1.py file

# =========== 7.1. Evaluating ARIMA model to get best order values ============

# df_result dataframe is defined to store the results at the end
df_result = pd.DataFrame()

# Looping the evalutions of ARIMA model for each account
for ind, eachact in actcount.items():

    # Getting all the rows for each account from the dataset
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period 
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M')    # [2]
    
    # ================ 7.2. Split the data in train and test  =================

    # Calling SplitTrainTestData function to split the data in train and test data
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ========================= 7.3. TUNE ARIMA Model =========================
    
    # Calling EvaluateTuneARIMAmodels function to evaluate ARIMA Model with various p,d,q values
    df_bestarima = EvaluateTuneARIMAModels(df_train, df_test, ind, p_values, d_values, q_values)
    # Storing the results [3] 
    df_result = pd.concat([df_result,df_bestarima], keys=["ACCOUNT_NO", "ORDER", "RMSE"], ignore_index=True)
    
    # Give a new line to clearly format the output
    Newline()
    
# ==================== 7.4. Plot RMSE of all accounts =========================

print("Best order with its RMSE for all accounts:")
print(df_result)
df_result.plot(kind='bar',figsize=(8,4))
plt.show()

# Give a new line to clearly format the output
Newline()

#Output:
#Best order with its RMSE for all accounts:
#     ACCOUNT_NO      ORDER      RMSE
#0       1196428  [1, 0, 5]  0.025024
#1       1196711  [5, 0, 5]  0.875718
#2  409000362497  [5, 0, 4]  0.285252
#3  409000405747  [4, 0, 3]  0.149556
#4  409000425051  [2, 0, 3]  0.011253
#5  409000438611  [5, 0, 1]  0.094484
#6  409000438620  [1, 0, 5]  0.000231
#7  409000493201  [5, 0, 5]  0.016095
#8  409000493210  [5, 0, 5]  0.000328
#9  409000611074  [3, 0, 5]  0.032077

######################## STEP 8: FINAL ARIMA MODEL ############################

start_time = datetime.now()
#print("Final ARIMA model start_time is - ", start_time)

# Clear the list
rmselist.clear()

# =========== 8.1. Evaluating ARIMA model to get best order values ============

# Evaluating ARIMA Model with the best order of p,d,q found after tuning the model

# Looping the evalutions of ARIMA model for each account
for ind, eachact in actcount.items():

    # Getting all the rows for each account from the dataset
    dfacctmaxrecords = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['TRANSACTION_AMOUNT']]
    # Taking only TRANSACTION_AMOUNT attribute into new dataframe to be worked upon further
    df_transdata = dfacctmaxrecords['TRANSACTION_AMOUNT']
    df_transdata = pd.DataFrame(df_transdata, columns = ['TRANSACTION_AMOUNT'])
    # There are many duplicate date entries in index, hence converting the index to monthly period 
    df_transdata.index = pd.DatetimeIndex(df_transdata.index).to_period('M')    # [2]
    
    # =============== 8.2. Split the data in train and test  ==================
    
    # Calling SplitTrainTestData function to split the data in train and test arrays
    trainingset, testset, df_train, df_test = SplitTrainTestData(df_transdata)
    print("For Account {} Training set is {} and Test set is {}".format(ind, trainingset.shape, testset.shape))
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================= 8.3. Retreive p,d,q values ======================
    
    # Get the p, d, q values out of the ORDER column in dataframe
    order = df_result.loc[df_result['ACCOUNT_NO']==str(ind) , ['ORDER']]
    ordervalues = order.iloc[0]
    p = ordervalues.iloc[0][0]
    d = ordervalues.iloc[0][1]
    q = ordervalues.iloc[0][2]
    
    # ====================== 8.4. Evaluate ARIMA Model ========================
        
    # Calling EvaluateARIMAModel function to train and predict basic ARIMA model with p,d,q
    testrmse, df_testpredictions = EvaluateARIMATestModel(df_train, df_test, p, d, q, ind)
    futurermse, df_futurepredictions = EvaluateARIMAFutureModel(df_train, df_test, p, d, q, ind)
    
    # Give a new line to clearly format the output
    Newline()
    
    # ======================== 8.5. Plot Predictions ==========================
    
    # Seting the plot size
    plt.rcParams["figure.figsize"] = (8,4)
    # Plot the ARIMA Predictions
    PlotTestFuturePrediction(df_test,df_testpredictions,df_futurepredictions,ind)
    
    # =========================== 8.6. Store RMSE =============================
    
    # Store Root Mean Squared Error for each Account
    rmseline = {'ACCOUNT_NO': str(ind), 'TESTRMSE': testrmse, 'FUTURERMSE': futurermse}
    rmselist.append(rmseline)

# ==================== 8.7. Plot RMSE of all accounts =========================

# Create dataframe containing ACCOUNT_NO with its RMSE 
df_finalrmse = pd.DataFrame(rmselist)
print("ARIMA Model: Accounts with its RMSE")
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
print("Total time to run the Final ARIMA model for all accounts is {}".format(totaltime))

# Output
#ARIMA Model: Accounts with its RMSE
#     ACCOUNT_NO  TESTRMSE  FUTURERMSE
#0       1196428  0.025024    0.025024
#1       1196711  0.875718    0.875718
#2  409000362497  0.285252    0.285252
#3  409000405747  0.149556    0.149556
#4  409000425051  0.011253    0.011253
#5  409000438611  0.094484    0.094484
#6  409000438620  0.000231    0.000231
#7  409000493201  0.016095    0.016095
#8  409000493210  0.000328    0.000328
#9  409000611074  0.032077    0.032077

# Total time to run the Final ARIMA model for all accounts is 0:06:20.620152


################################# References ##################################

# [1] https://stackoverflow.com/questions/72336200/deprecated-arma-module
# [2] https://stackoverflow.com/questions/58510659/error-valuewarning-a-date-index-has-been-provided-but-it-has-no-associated-fr
# [3] https://pandas.pydata.org/docs/user_guide/merging.html



