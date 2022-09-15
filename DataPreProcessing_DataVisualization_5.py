#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:40:29 2022

Student ID: 29837043
File: DataPreProcessing_DataVisualization_5.py

This file contains the Exploratory data analysis of dataset to make the data in readable format for 
data visualization and modelling. Also contains the Data Visualization on the transformed dataset.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import sklearn
# Import train_test_split function
from sklearn.model_selection import train_test_split
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
import statistics
from statistics import mode

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

########################### STEP 2: READING DATA ##############################


# Calling ReadDataset function from FunctionLibrary.py file. 
# It reads the dataset file and store it in a dataframe along with its attributes
# location_of_file is defined in the Config.py file
# The dataset is handled for the duplicate date values by adding the ms units to date
# DATE attribute is set as index and then data is sorted
dfbank = ReadDataset(location_of_file, originalfile)

# Print the number of rows and columns present in dataset
print("The dataset has Rows {} and Columns {} ".format(dfbank.shape[0], dfbank.shape[1]))

# Give a new line to clearly format the output
Newline()

# Output: 
# The dataset has Rows 116201 and Columns 8 

##################### STEP 3: PRE-PROCESSING THE DATA #########################

#======================== 3.1. Understand the data ============================

# Print the information of dataframe
print(dfbank.info())

# Output: As observed the dataset has a "." column, which will be removed in further steps

#<class 'pandas.core.frame.DataFrame'>
#DatetimeIndex: 116201 entries, 2015-01-01 00:00:00 to 2019-03-05 00:00:00.110000
#Data columns (total 8 columns):
# #   Column               Non-Null Count   Dtype         
#---  ------               --------------   -----         
# 0   ACCOUNT_NO           116201 non-null  object        
# 1   TRANSACTION_DETAILS  113702 non-null  object        
# 2   CHQ.NO.              905 non-null     float64       
# 3   VALUE_DATE           116201 non-null  datetime64[ns]
# 4   WITHDRAWAL_AMT       53549 non-null   float64       
# 5   DEPOSIT_AMT          62652 non-null   float64       
# 6   BALANCE_AMT          116201 non-null  float64       
# 7   .                    116201 non-null  object        
#dtypes: datetime64[ns](1), float64(4), object(3)
#memory usage: 8.0+ MB
#None

# Give a new line to clearly format the output
Newline()
    
# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset")
print(dfbank.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns CHQ.NO., WITHDRAWAL_AMT, DEPOSIT_AMT, BALANCE_AMT, which specifies that 
# other columns are having some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

# Output: As observed, below data is showing huge mean and std. 
# This will be taken care during standardization and detecting ourliers

#Summary of dataset
#             CHQ.NO.  WITHDRAWAL_AMT   DEPOSIT_AMT   BALANCE_AMT
#count     905.000000    5.354900e+04  6.265200e+04  1.162010e+05
#mean   791614.503867    4.489190e+06  3.806586e+06 -1.404852e+09
#std    151205.932910    1.084850e+07  8.683093e+06  5.348202e+08
#min         1.000000    1.000000e-02  1.000000e-02 -2.045201e+09
#25%    704231.000000    3.000000e+03  9.900000e+04 -1.690383e+09
#50%    873812.000000    4.708300e+04  4.265000e+05 -1.661395e+09
#75%    874167.000000    5.000000e+06  4.746411e+06 -1.236888e+09
#max    874525.000000    4.594475e+08  5.448000e+08  8.500000e+06


#======================= 3.2. Find the missing values =========================

# Printing the missing values (NaN) [1]
print("Attributes having missing values")
print(dfbank.isnull().sum())

# Give a new line to clearly format the output
Newline()

# The output below shows that there are missing values (NaN) avaliable in the dataset
# The rows with missing values in TRANSACTION_DETAILS attribute, will be udpated in step 3.3 and 3.8
# WITHDRAWAL_AMT and DEPOSIT_AMT attributes, will be filled with 0 in the step 3.3
# Also, CHQ.NO. and "." attributes are not needed for this research, hence it will be removed from the dataframe in step 3.5

# Output:
# Attributes having missing values
#ACCOUNT_NO                  0
#TRANSACTION_DETAILS      2499
#CHQ.NO.                115296
#VALUE_DATE                  0
#WITHDRAWAL_AMT          62652
#DEPOSIT_AMT             53549
#BALANCE_AMT                 0
#.                           0
#dtype: int64

#================ 3.3 Fill the empty columns with some values =================

# Fill the empty DEPOSIT_AMT, WITHDRAWAL_AMT and TRANSACTION_DETAILS columns with zero and specific categorical value
FillEmptyColumn(dfbank,'DEPOSIT_AMT',0)
FillEmptyColumn(dfbank,'WITHDRAWAL_AMT',0)
FillEmptyColumn(dfbank,'TRANSACTION_DETAILS','NoTransDetail')

# Wait until the graph is displayed
time.sleep(20)
    
#=========================== 3.4 Add new attribute ============================

# Adding new attribute TRANSACTION_AMOUNT to have the information of transactions happened
dfbank['TRANSACTION_AMOUNT'] = (dfbank['DEPOSIT_AMT'] - dfbank['WITHDRAWAL_AMT'])

#=================== 3.5 Dropping unwanted attributes =========================

# Droppping CHQ.NO. and "." attributes as its not needed for the analysis and stored it in a new dataframe
dfbankdata = dfbank[['ACCOUNT_NO', 'TRANSACTION_DETAILS', 'WITHDRAWAL_AMT','DEPOSIT_AMT','TRANSACTION_AMOUNT', 'BALANCE_AMT']]


# Print the number of rows and columns present in dataset
print("The dataset has Rows {} and Columns {} ".format(dfbankdata.shape[0], dfbankdata.shape[1]))

# Give a new line to clearly format the output
Newline()

# Output:
# The dataset has Rows 116201 and Columns 6 

#=================== 3.6 Remove the special characters ========================

# Calling RemoveSpecialCharacter function to remove special character "'" from ACCOUNT_NO attribute
RemoveSpecialCharacter(dfbankdata,"ACCOUNT_NO",specialchar)

#=================== 3.7 Convert the datatype to string =======================

# Check the datatype of TRANSACTION_DETAILS attribute
print("The datatype of TRANSACTION_DETAILS is ", type(dfbankdata['TRANSACTION_DETAILS']))

# Give a new line to clearly format the output
Newline()

# Output:
# The datatype of TRANSACTION_DETAILS is  <class 'pandas.core.series.Series'>

# Converting datatype of TRANSACTION_DETAILS attribute
dfbankdata = dfbankdata.astype({"TRANSACTION_DETAILS": str})

# Print the information of dataframe
print(dfbankdata.info())

# Give a new line to clearly format the output
Newline()

# Output:
#<class 'pandas.core.frame.DataFrame'>
#DatetimeIndex: 116201 entries, 2015-01-01 00:00:00 to 2019-03-05 00:00:00.110000
#Data columns (total 6 columns):
# #   Column               Non-Null Count   Dtype  
#---  ------               --------------   -----  
# 0   ACCOUNT_NO           116201 non-null  object 
# 1   TRANSACTION_DETAILS  116201 non-null  object 
# 2   WITHDRAWAL_AMT       116201 non-null  float64
# 3   DEPOSIT_AMT          116201 non-null  float64
# 4   TRANSACTION_AMOUNT   116201 non-null  float64
# 5   BALANCE_AMT          116201 non-null  float64
#dtypes: float64(4), object(2)
#memory usage: 6.2+ MB
#None

#===================== 3.8 Update the category values =========================

# There are many transaction details provided in the dataset, which are updated with common categories to mimic the Merchant Category.

# Calling ReplaceCategory function to update the TRANSACTION_DETAILS values to a standard value and store it in new dataframe
dfbankdataset = ReplaceCategory(dfbankdata,"TRANSACTION_DETAILS",categorylist)
# Calling ReplaceIntegerCategory to replace the reference numbers with RefNum as Category
dfbankdataset = ReplaceIntegerCategory(dfbankdataset,"TRANSACTION_DETAILS")
# Calling ReplaceIndividualTrans to replace the individuals transactions 
dfbankdataset = ReplaceIndividualTrans(dfbankdataset,"TRANSACTION_DETAILS",categorylist)

# Print the number of rows and columns present in dataset
print("The categorized dataset has Rows {} and Columns {} ".format(dfbankdataset.shape[0], dfbankdataset.shape[1]))

# Give a new line to clearly format the output
Newline()

# Output: 
# The categorized dataset has Rows 116201 and Columns 6

#=============== 3.9 Draw Pie chart to show the transactions  =================

# Calling PlotPiechartPerAccount to plot pie chart for each account to show the TRANSACTION_DETAILS against WITHDRAWAL_AMT and DEPOSIT_AMT
PlotPiechartPerAccount(dfbankdataset, 'WITHDRAWAL_AMT')
PlotPiechartPerAccount(dfbankdataset, 'DEPOSIT_AMT')

#====================== 3.10 Standardize the attributes =======================

# The amount provided in dataset has high numbers, which becomes difficult to interpret it in visualization and machine learning algorithms. 
# Attributes are scaled with Standardization technique so that the mean reduces and standard deviation becomes 1.

# Calling StdData function to standardize the float values in dataframe
df_toscale = dfbankdataset[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT', 'TRANSACTION_AMOUNT']]
scaleddata, scaler = StdData(df_toscale)
df_scaled = pd.DataFrame(scaleddata, columns = ['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT', 'TRANSACTION_AMOUNT'])
df_scaled.index = dfbankdataset.index
dfbankdataset[['WITHDRAWAL_AMT','DEPOSIT_AMT','BALANCE_AMT', 'TRANSACTION_AMOUNT']] = df_scaled[['WITHDRAWAL_AMT','DEPOSIT_AMT','BALANCE_AMT', 'TRANSACTION_AMOUNT']]

#======================== 3.11. Understand the data ===========================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset after standardization")
print(dfbankdataset.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# for columns WITHDRAWAL_AMT, DEPOSIT_AMT, TRANSACTION_AMOUNT, BALANCE_AMT. 
# The data looks properly standardised as it shows smaller mean value and std close to 1

# Give a new line to clearly format the output
Newline()

# Output:
#Summary of dataset after standardization
#       WITHDRAWAL_AMT   DEPOSIT_AMT  TRANSACTION_AMOUNT   BALANCE_AMT
#count    1.162010e+05  1.162010e+05        1.162010e+05  1.162010e+05
#mean     5.004066e-14 -1.223258e-13       -1.636317e+04 -2.764209e-15
#std      1.000004e+00  1.000004e+00        1.058230e+07  1.000004e+00
#min     -2.687798e-01 -3.085316e-01       -4.594475e+08 -1.197322e+00
#25%     -2.687798e-01 -3.085316e-01       -2.682500e+04 -5.338852e-01
#50%     -2.687798e-01 -3.077799e-01        5.000000e+03 -4.796835e-01
#75%     -2.652946e-01 -2.333677e-01        5.000000e+05  3.140576e-01
#max      5.942417e+01  8.158998e+01        5.448000e+08  2.642679e+00

####################### Step 4: DATA VISUALIZATION ############################

#============================ 4.1. Plot Heat map ==============================

# To visualize the data, heat map is demonstrated below
# Heat map is created with WITHDRAWAL_AMT, DEPOSIT_AMT, TRANSACTION_AMOUNT and BALANCE_AMT
# Calling DrawHeatmap function to create the Heat map
DrawHeatmap(dfbankdataset)

# Wait until the graph is displayed
time.sleep(10)

# As observed in the heat plot, the data has high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes. 

#============================ 4.2. Plot Subplots ==============================

# To visualize the data again, Subplot is shown below
# Calling SubPlotsforAmt to show the spikes for WITHDRAWAL_AMT, DEPOSIT_AMT and BALANCE_AMT
SubPlotsforAmt(dfbankdataset)

# Wait until the graph is displayed
time.sleep(10)

# According to the plots shown, the amount withdrawn spike is higher in around 20000, 60000, 90000 and just below 1100000
# whereas the deposit amount does not show any spike around 90000, but there are 2 spikes near 60000
# also the the balance amount is decreasing eventually

#============================= 4.3. Plot Boxplot ==============================

# To visualize the pattern of WITHDRAWAL_AMT, DEPOSIT_AMT and BALANCE_AMT boxplot is demostrated below
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","WITHDRAWAL_AMT")
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","DEPOSIT_AMT")
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","TRANSACTION_AMOUNT")

# Wait until the graph is displayed
time.sleep(10)

# As observed in the Boxplot, the quantile range is pretty small and there are many data points above and below the quantile
# These point can be outliers and hence the outliers are handled in step 4.4.

#==================== 4.4. Detecting Outliers in data ==========================

# From the above plots it is clear that there are a lot of outliers.
# Many data points are falling outside the 25% and 75% quantile
# Therefore, detecting the outliers and replace it using imputation as if they were missing values
# Getting the number of rows for each account 
actcount = dfbankdataset.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()

# Looping the evalutions of outliers for each account 
for ind, eachact in actcount.items():
    dfacctmaxrecords = pd.DataFrame()
    # Getting all the rows for each account from the dataset
    dfacctmaxrecords[['WITHDRAWAL_AMT','DEPOSIT_AMT','TRANSACTION_AMOUNT']] = dfbankdataset.loc[dfbankdataset['ACCOUNT_NO']==ind, ['WITHDRAWAL_AMT','DEPOSIT_AMT', 'TRANSACTION_AMOUNT']]
    # Calling FindOutliers_IQR funciton to find the outliers in "WITHDRAWAL_AMT", "DEPOSIT_AMT" and "TRANSACTION_AMOUNT" attributes
    outliers_wa, correcteddataset = FindOutliers_IQR(dfacctmaxrecords)
    # Converting the corrected dataset into dataframe
    dfdataset = pd.DataFrame(correcteddataset, columns = ['WITHDRAWAL_AMT','DEPOSIT_AMT', 'TRANSACTION_AMOUNT'])
    dfdataset.index = dfacctmaxrecords.index
    # Replace the corrected values into original dataframe [] https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
    dfbankdataset.loc[dfbankdataset.index.isin(dfdataset.index), ['WITHDRAWAL_AMT','DEPOSIT_AMT', 'TRANSACTION_AMOUNT']] = dfdataset[['WITHDRAWAL_AMT','DEPOSIT_AMT', 'TRANSACTION_AMOUNT']]
    
# Output:
# 25% quantile is -0.26877975934372483
#75% quantile is -0.2652945665247695
#IQR is 0.0034851928189553183
#number of outliers: 25680
#max outlier value: 59.42416671821493
#min outlier value: -0.26006658241137

#25% quantile is -0.30853157964301237
#75% quantile is -0.2333677452746575
#IQR is 0.07516383436835486
#number of outliers: 21930
#max outlier value: 81.58998234811645
#min outlier value: -0.1206219937221252

#25% quantile is -0.0009886196194325164
#75% quantile is 0.048795178777463845
#IQR is 0.04978379839689636
#number of outliers: 41428
#max outlier value: 51.48394246189935
#min outlier value: -43.41523162544288

#======================= 4.5. Plot Boxplot per account ========================

# To visualize the pattern for each account after correcting the outliers, boxplot is demostrated below
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","WITHDRAWAL_AMT")
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","DEPOSIT_AMT")
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","TRANSACTION_AMOUNT")

# Wait until the graph is displayed
time.sleep(10)

# Looking at the boxplot for all accounts after correcting outliers,
# there are very few data points falling outside the range of 25 and 75 percentile quantiles

#==================== 4.6. Plot Monthly Data per account =======================

# Calling PlotYearlyDataPerAccount function to plot the monthly data for each account
PlotBusinessDataPerAccount(dfbankdataset,'WITHDRAWAL_AMT', 'M')
PlotBusinessDataPerAccount(dfbankdataset,'DEPOSIT_AMT', 'M')
PlotBusinessDataPerAccount(dfbankdataset,'TRANSACTION_AMOUNT', 'M')

# Wait until the graph is displayed
time.sleep(10)

##################### STEP 5: FEATURE SELECTION PREP ##########################

#======= 5.1. Analysing and making consistent data for an attributes ==========

# Convert the column values from string/object into integer using labelencoder    
# Calling the Encoder function to convert the categorical values to integer in the dataframe
dfbankdataset = Encoder(dfbankdataset, ['TRANSACTION_DETAILS'])

# Give a new line to clearly format the output
Newline()

#======================== 5.2. Class distribution =============================

# "TRANSACTION_DETAILS" attribute is taken as the class attribute

# Finding the number of records that belong to each class
print(dfbankdataset.groupby('TRANSACTION_DETAILS').size())

# Give a new line to clearly format the output
Newline()

# Output: There are 93 classes in TRANSACTION_DETAILS after encoding the categories
# TRANSACTION_DETAILS
#0       19
#1     3546
#2       14
#3     3821
#4        9
#      ... 
#88       8
#89       9
#90     239
#91     842
#92      38
#Length: 93, dtype: int64

#======================== 5.3. Understand the data ============================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset")
print(dfbankdataset.describe())

# Save the cleaned dataset in xlsx file
dfbankdataset.to_excel(cleanfile)

# Give a new line to clearly format the output
Newline()

# Output:
#Summary of dataset
#       TRANSACTION_DETAILS  WITHDRAWAL_AMT   DEPOSIT_AMT  TRANSACTION_AMOUNT    BALANCE_AMT
#count        116201.000000    1.162010e+05  1.162010e+05        1.162010e+05   1.162010e+05 
#mean             53.643876   -2.090731e-01 -2.364404e-01        6.969018e+04  -2.764209e-15 
#std              27.828696    1.113631e-01  1.183131e-01        2.399629e+05   1.000004e+00 
#min               0.000000   -2.687798e-01 -3.085316e-01       -8.168320e+05  -1.197322e+00 
#25%              26.000000   -2.687798e-01 -3.085316e-01       -1.636317e+04  -5.338852e-01 
#50%              47.000000   -2.687798e-01 -3.077799e-01       -6.970000e+03  -4.796835e-01 
#75%              85.000000   -2.652946e-01 -2.333677e-01        5.000000e+04   3.140576e-01 
#max              92.000000    5.004066e-14 -1.223258e-13        1.290068e+06   2.642679e+00

# The dataset has been cleaned and saved into an excel sheet, which will be read for further implementation


################################# References ##################################

# [1] https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe

