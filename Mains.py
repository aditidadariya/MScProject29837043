#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: mainsfile.py

This file contains the implementation of models. 
Variables declared in config.py file will be used here
Functions declared in functionlibrary.py file will be called here to work on the algorithms

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
# import seaborn as sns
# Import matplotlib.pyplot to draw and save plots
# import matplotlib.pyplot as plt

# Import Decision Tree Classifier
#from sklearn.tree import DecisionTreeClassifier
# Import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Import Logistic Regression
#from sklearn.linear_model import LogisticRegression
# Import KNeighbours Classifier
#from sklearn.neighbors import KNeighborsClassifier
# Import Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
# Import Support Vector Machine
#from sklearn.svm import SVC

# Import LinearRegression
#from sklearn.linear_model import LinearRegression
# Import RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor
# Import DecisionTreeRegressor
#from sklearn.tree import DecisionTreeRegressor
# Import Ridge
#from sklearn.linear_model import Ridge
# Import Lasso
#from sklearn.linear_model import Lasso
# Import SGDRegressor
#from sklearn.linear_model import SGDRegressor
# Import KNeighborsRegressor
#from sklearn.neighbors import KNeighborsRegressor
# Import LinearSVR
#from sklearn.svm import LinearSVR
# Import SVR
#from sklearn.svm import SVR


# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV function
from sklearn.model_selection import GridSearchCV
# Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics 
# Import confusion_matrix function
#from sklearn.metrics import confusion_matrix
# Import classification_report function
#from sklearn.metrics import classification_report
# Import ConfusionMatrixDisplay function to plot confusion matrix
#from sklearn.metrics import ConfusionMatrixDisplay
# Import render from graphviz to convert dot file to png
from graphviz import render
# Import RandomUnderSampler, SMOTE, Pipeline to balance the dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from datetime import datetime
import time
# Import Config.py file
from Config import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess import *
# Import FuncLibVisual.py file
from FuncLibVisual import *
# Import FuncLibModel.py file
from FuncLibModel import *

# Import mean_squared_error and r2_score
#from sklearn.metrics import mean_squared_error, r2_score

########################### STEP 2: READING DATA ##############################


# Calling ReadDataset function from FunctionLibrary.py file. 
# It reads the dataset file [1] and store it in a dataframe along with its attributes
# location_of_file is defined in the Config.py file
dfbank = ReadDataset(location_of_file)


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

# Output:
#<class 'pandas.core.frame.DataFrame'>
#DatetimeIndex: 116201 entries, 2017-06-29 to 2019-03-05
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
    
# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset")
print(dfbank.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns CHQ.NO., WITHDRAWAL_AMT, DEPOSIT_AMT, BALANCE_AMT, which specifies that 
# other columns are having some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

# Output:
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

# Printing the missing values (NaN)
print("Attributes having missing values")
print(dfbank.isnull().sum())

# Give a new line to clearly format the output
Newline()

# The output below shows that there are missing values (NaN) avaliable in the dataset
# The rows with missing values in TRANSACTION_DETAILS attribute, will be dropped in step 3.3
# and WITHDRAWAL_AMT and DEPOSIT_AMT attributes, will be filled with 0 in the step 3.4
# Also, CHQ.NO. and "." attributes are not needed for this research, hence it will be removed from the dataframe in step 3.6

# Output:
# Attributes having missing values
# Account_No                  0
# TRANSACTION_DETAILS      2499
# CHQ.NO.                115296
# VALUE_DATE                  0
# WITHDRAWAL_AMT          62652
# DEPOSIT_AMT             53549
# BALANCE_AMT                 0
# .                           0
# dtype: int64


### SECTION NOT NEEDED ==================
##========================== 3.3 Remove empty rows =============================

## Drop the empty rows in TRANSACTION_DETAILS attribute []
#dfbank.dropna(subset=['TRANSACTION_DETAILS'], inplace=True)
## Printing the missing values (NaN)
#print("Attributes having missing values")
#print(dfbank.isnull().sum())

## Give a new line to clearly format the output
#Newline()

## Output:
## Attributes having missing values
## ACCOUNT_NO                  0
## DATE                        0
## TRANSACTION_DETAILS         0
## CHQ.NO.                112797
## VALUE_DATE                  0
## WITHDRAWAL_AMT          60153
## DEPOSIT_AMT             53549
## BALANCE_AMT                 0
## .                           0
## dtype: int64

## Reference: https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe


#================ 3.3 Fill the empty columns with some values =================

# Fill the empty DEPOSIT_AMT, WITHDRAWAL_AMT and TRANSACTION_DETAILS columns with zero and specific value
FillEmptyColumn(dfbank,'DEPOSIT_AMT',0)
FillEmptyColumn(dfbank,'WITHDRAWAL_AMT',0)
FillEmptyColumn(dfbank,'TRANSACTION_DETAILS','NoTransDetail')

# Wait until the graph is displayed
time.sleep(20)
    
#=========================== 3.4 Add new attribute ============================

# Adding new attribute TRANSACTION_AMOUNT to have the information of transactions happened
dfbank['TRANSACTION_AMOUNT'] = (dfbank['DEPOSIT_AMT'] - dfbank['WITHDRAWAL_AMT'])

# Print few rows of dataframe
#print(dfbank.head())

# Give a new line to clearly format the output
#Newline()

#=================== 3.5 Dropping unwanted attributes =========================

# Droppping CHQ.NO. and "." attributes as its not needed for the analysis and stored it in a new dataframe
#dfbankdata = dfbank[['ACCOUNT_NO', 'DATE', 'TRANSACTION_DETAILS', 'VALUE_DATE', 'WITHDRAWAL_AMT','DEPOSIT_AMT','TRANSACTION_AMOUNT', 'BALANCE_AMT']]
dfbankdata = dfbank[['ACCOUNT_NO', 'TRANSACTION_DETAILS', 'WITHDRAWAL_AMT','DEPOSIT_AMT','TRANSACTION_AMOUNT', 'BALANCE_AMT']]


# Print the number of rows and columns present in dataset
print("The dataset has Rows {} and Columns {} ".format(dfbankdata.shape[0], dfbankdata.shape[1]))

# Give a new line to clearly format the output
Newline()

# Output:
# The dataset has Rows 116201 and Columns 6 

#=================== 3.6 Remove the special characters ========================

# RemoveSpecialCharacter function defined to remove the special character "'" from ACCOUNT_NO attribute
#dfbankdata['ACCOUNT_NO'] = dfbankdata['ACCOUNT_NO'].replace("'", '',regex=True)
#dfbankdata.ACCOUNT_NO = dfbankdata.ACCOUNT_NO.str.replace("'", '',regex=True)

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

# Calling ConvertDataTypeStr to convert the datatype of TRANSACTION_DETAILS to string
#ConvertDataTypeStr(dfbankdata, "TRANSACTION_DETAILS")

# Print the information of dataframe
print(dfbankdata.info())

#print(dfbankdata.head(5))

# Give a new line to clearly format the output
Newline()

# Output:
#<class 'pandas.core.frame.DataFrame'>
#DatetimeIndex: 116201 entries, 2017-06-29 to 2019-03-05
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


#===================== 3.8 Update the category values =========================

# Update the TRANSACTION_DETAILS values to a standard value and store it in new dataframe
# by calling ReplaceCategory function
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
#The categorized dataset has Rows 116201 and Columns 6

### SECTION NOT NEEDED ==================
##============== 3.9 Drop rows where TRANSACTION_DETAILS unwanted ==============

## Calling DropRows to drop unwanted rows based on TRANSACTION_DETAILS attribute
##DropRows(dfbankdataset)
## Print the number of rows and columns present in dataset
#print("The dataset has Rows {} and Columns {} after categorization".format(dfbankdataset.shape[0], dfbankdataset.shape[1]))

## Give a new line to clearly format the output
#Newline()

## Outout:
## The dataset has Rows 113754 and Columns 8 after categorization

## The dataset has Rows 116201 and Columns 6 after categorization # (This is after DATE as Index)


#====================== 3.10 Standardize the attributes =======================

# Calling StdData function to standardize the float values in dataframe
StdData(dfbankdataset)
# Display few rows
#print(dfbankdataset.head(5))

# Give a new line to clearly format the output
#Newline()

# Output: 
#     ACCOUNT_NO       DATE TRANSACTION_DETAILS VALUE_DATE  WITHDRAWAL_AMT  DEPOSIT_AMT  TRANSACTION_AMOUNT  BALANCE_AMT 
#0  409000611074 2017-06-29       Indiaforensic 2017-06-29       -0.278867     -0.165418            0.096736     2.657779
#1  409000611074 2017-07-05       Indiaforensic 2017-07-05       -0.278867     -0.165418            0.096736     2.659668 
#2  409000611074 2017-07-18                 Dr. 2017-07-18       -0.278867     -0.241399            0.048753     2.660613
#3  409000611074 2017-08-01       Indiaforensic 2017-08-01       -0.278867      0.138507            0.288667     2.666281
#4  409000611074 2017-08-16                 Dr. 2017-08-16       -0.278867     -0.241399            0.048753     2.667225


#======================== 3.11. Understand the data ===========================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset after standardization")
print(dfbankdataset.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns CHQ.NO., WITHDRAWAL_AMT, DEPOSIT_AMT, BALANCE_AMT, which specifies that 
# other columns are having some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

# Output:
#Summary of dataset after standardization
#       WITHDRAWAL_AMT   DEPOSIT_AMT  TRANSACTION_AMOUNT   BALANCE_AMT
#count    1.137540e+05  1.137540e+05        1.137540e+05  1.137540e+05
#mean    -2.222676e-14 -9.964335e-14       -7.174218e-17 -4.533437e-15
#std      1.000004e+00  1.000004e+00        1.000004e+00  1.000004e+00
#min     -2.788674e-01 -3.173805e-01       -4.409056e+01 -1.207934e+00
#25%     -2.788674e-01 -3.173805e-01       -1.976027e-03 -5.373761e-01
#50%     -2.788674e-01 -3.167726e-01        1.153423e-03 -4.820963e-01
#75%     -2.750619e-01 -2.364605e-01        5.187143e-02  3.247541e-01
#max      6.083299e+01  8.247176e+01        5.228302e+01  2.671948e+00

# Output after DATE Indexed:
#Summary of dataset after standardization
#       WITHDRAWAL_AMT   DEPOSIT_AMT  TRANSACTION_AMOUNT   BALANCE_AMT
#count    1.162010e+05  1.162010e+05        1.162010e+05  1.162010e+05
#mean    -4.588502e-14 -1.202224e-14       -1.116734e-16  1.423286e-15
#std      1.000004e+00  1.000004e+00        1.000004e+00  1.000004e+00
#min     -2.687798e-01 -3.085316e-01       -4.341523e+01 -1.197322e+00
#25%     -2.687798e-01 -3.085316e-01       -9.886196e-04 -5.338852e-01
#50%     -2.687798e-01 -3.077799e-01        2.018773e-03 -4.796835e-01
#75%     -2.652946e-01 -2.333677e-01        4.879518e-02  3.140576e-01
#max      5.942417e+01  8.158998e+01        5.148394e+01  2.642679e+00


####################### Step 4: DATA VISUALIZATION ############################


### SECTION NOT NEEDED ==================
##=========================== 4.1. Plot Pairplot ===============================

## To visualize the data, pair plot has been demonstrated below [6]
#DrawPairplot(dfbankdata)

## The plots suggests that there is a high correlation and a predictable relationship 
## between the diagonal grouping of some pairs of attributes
## As stated earlier that the data is imbalanced, the graph is not clearly linear too.

#============================ 4.2. Plot Heat map ==============================

# To visualize the data again, heat map is demonstrated below [7]
DrawHeatmap(dfbankdata)

# As observed earlier in the pairplot, the data has high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes. 
# The same has been observed in Heat map as well.
# To fix the imbalance problem, I have used the balancing technique RandomUnderSampler and SMOTE
# in the further steps after checking the performance of Decision Tree and Linear Discriminent Analysis models
# using the basic HoldOut Validation (i.e. train and test split)

#============================= 4.3. Plot Boxplot ==============================

# To visualize the pattern of WITHDRAWAL_AMT, DEPOSIT_AMT and BALANCE_AMT boxplot is demostrated below []
#DrawBoxplot(dfbankdata)
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","WITHDRAWAL_AMT")
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","DEPOSIT_AMT")
PlotDataByAcct(dfbankdataset,"ACCOUNT_NO","BALANCE_AMT")

# As observed in the Boxplot, out of 10 accounts, 2 accounts do not have any withdrawal 
# and 1 account does not have any deposit
# Also, the balance of 3 account 

# References: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html

#============================= 4.4. Plot Boxplot per account ==============================

# To visualize the pattern for each account, boxplot is demostrated below
#BoxPlotPerAcct(dfbankdataset)
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","WITHDRAWAL_AMT")
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","DEPOSIT_AMT")
PlotDataPerAcctCat(dfbankdataset,"TRANSACTION_DETAILS","BALANCE_AMT")

# Looking at the boxplot for all accounts,


### SECTION NOT NEEDED ==================
##============================= 4.4. Plot Boxplot per account ==============================

## To visualize the pattern for each account, boxplot is demostrated below
##PlotData(dfbankdataset,"ACCOUNT_NO","WITHDRAWAL_AMT")
#PlotData(dfbankdataset,"TRANSACTION_DETAILS","WITHDRAWAL_AMT")

##PlotData(dfbankdataset,"ACCOUNT_NO","DEPOSIT_AMT")
#PlotData(dfbankdataset,"TRANSACTION_DETAILS","DEPOSIT_AMT")

##PlotData(dfbankdataset,"ACCOUNT_NO","BALANCE_AMT")
#PlotData(dfbankdataset,"TRANSACTION_DETAILS","BALANCE_AMT")

## Looking at the boxplot for all accounts,


##################### STEP 5: FEATURE SELECTION PREP ##########################

#==================== 5.1 Convert the datatype of datetime to str =============

# Converting datatype of DATE and VALUE_DATE attributes from Timestamp to String
#dfbankdataset['DATE'] = dfbankdataset[('DATE')].values.astype("float64")
#dfbankdataset['VALUE_DATE'] = dfbankdataset[('VALUE_DATE')].values.astype("float64")

#======= 5.2 Analysing and making consistent data for an attributes ===========

# Convert the column values from string/object into integer using labelencoder [5]

# In the dataset, there are mix the data types, such as values as string, float and integer
# To have same datatype, created Encoder function to convert the strings into integer values
     
# Calling the Encoder function passing the dataframe as parameter that creats a new dataframe to store the new values
dfbankdataset = Encoder(dfbankdataset)

# Save the new dataframe into a csv file
#dfbankdataset =  pd.DataFrame(dfbankdataset)
# Save the dataset in .csv file
dfbankdataset.to_csv('bankdatafinal.csv')

# Give a new line to clearly format the output
Newline()

# Summarizing the data statisticaly again by getting the mean, std and other values on all the columns
#print(dfbankdata.head())

# Give a new line to clearly format the output
#Newline()

#======================== 5.3 Class distribution =============================

# "TRANSACTION_DETAILS" attribute is taken as the class attribute

# Finding the number of records that belong to each class
print(dfbankdataset.groupby('TRANSACTION_DETAILS').size())

# Output states that the data is imbalanced. 
# To understand the dataset more, Data Visualization by ploting the pair plot and heat graph is done in next step

# Output:
# TRANSACTION_DETAILS
#0       19
#1     3546
#2       14
#3     3821
#4        9
#
#88       8
#89       9
#90     239
#91     842
#92      38
#Length: 93, dtype: int64

# Give a new line to clearly format the output
Newline()

#======================== 5.4 Understand the data =============================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print("Summary of dataset")
print(dfbankdataset.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns CHQ.NO., WITHDRAWAL_AMT, DEPOSIT_AMT, BALANCE_AMT, which specifies that 
# other columns are having some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

# Output:
#Summary of dataset
#               DATE  TRANSACTION_DETAILS    VALUE_DATE  WITHDRAWAL_AMT    DEPOSIT_AMT  TRANSACTION_AMOUNT   BALANCE_AMT
#count  1.137540e+05        113754.000000  1.137540e+05    1.137540e+05   1.137540e+05        1.137540e+05  1.137540e+05 
#mean   1.495300e+18            75.579030  1.495300e+18   -2.222676e-14  -9.964335e-14       -7.174218e-17 -4.533437e-15 
#std    3.481433e+16            49.785175  3.481433e+16    1.000004e+00   1.000004e+00        1.000004e+00  1.000004e+00 
#min    1.420070e+18             0.000000  1.420070e+18   -2.788674e-01  -3.173805e-01       -4.409056e+01 -1.207934e+00 
#25%    1.464134e+18            30.000000  1.464134e+18   -2.788674e-01  -3.173805e-01       -1.976027e-03 -5.373761e-01 
#50%    1.496621e+18            75.000000  1.496621e+18   -2.788674e-01  -3.167726e-01        1.153423e-03 -4.820963e-01 
#75%    1.527811e+18           103.000000  1.527811e+18   -2.750619e-01  -2.364605e-01        5.187143e-02  3.247541e-01 
#max    1.551744e+18           172.000000  1.551744e+18    6.083299e+01   8.247176e+01        5.228302e+01  2.671948e+00

# Output after DATE Indexed
# Summary of dataset
#       TRANSACTION_DETAILS  WITHDRAWAL_AMT   DEPOSIT_AMT  TRANSACTION_AMOUNT    BALANCE_AMT
#count        116201.000000    1.162010e+05  1.162010e+05        1.162010e+05   1.162010e+05
#mean            145.039208   -4.588502e-14 -1.200659e-14       -1.123508e-16   1.283451e-15
#std             108.207174    1.000004e+00  1.000004e+00        1.000004e+00   1.000004e+00
#min               0.000000   -2.687798e-01 -3.085316e-01       -4.341523e+01  -1.197322e+00
#25%              55.000000   -2.687798e-01 -3.085316e-01       -9.886196e-04  -5.338852e-01
#50%             130.000000   -2.687798e-01 -3.077799e-01        2.018773e-03  -4.796835e-01
#75%             199.000000   -2.652946e-01 -2.333677e-01        4.879518e-02   3.140576e-01
#max             370.000000    5.942417e+01  8.158998e+01        5.148394e+01   2.642679e+00


### SECTION NOT NEEDED ==================
##=========================== 5.5 Plot the data ================================

## Calling PlotData funciton to plot the final data grouped by ACCOUNT_NO against WITHDRAWAL_AMT,DEPOSIT_AMT and BALANCE_AMT
#PlotData(dfbankdataset,"ACCOUNT_NO","WITHDRAWAL_AMT")
#PlotData(dfbankdataset,"ACCOUNT_NO","DEPOSIT_AMT")
#PlotData(dfbankdataset,"ACCOUNT_NO","BALANCE_AMT")


####################### STEP 6: FEATURE SELECTION  ############################

# To implement any model on the data, we need to do feature selection first
# and therefore the dataset is devided into 2 parts.
# Firstly features (all the attributes except the target attribute) and secondly target (class attribute).
# The dfcredit dataframe has been split into dfcredittarget having only column A16, as this is the class attribute,
# and all the other attributes are taken into dfcreditdata dataframe as features

dfbankfeatures = dfbankdataset.drop("TRANSACTION_DETAILS", axis=1) # Features
# dfbanktarget = dfbankdata.iloc[:,-1]    # Target
dfbanktarget = dfbankdataset.loc[:, dfbankdataset.columns == "TRANSACTION_DETAILS"]     # Target []

# None of the feature selection methods are used here because the dataset is having very less number of attributes.
# and therefore, the dimentionality reduction has not been performed here.

# References:
# [] https://www.tutorialspoint.com/how-to-select-all-columns-except-one-in-a-pandas-dataframe#:~:text=To%20select%20all%20columns%20except%20one%20column%20in%20Pandas%20DataFrame,%5D.


"""# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ BUILDING BASIC MODELS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


############## STEP 6: BUILD BASE MODELS WITH HOLDOUT VALIDATION ###################

# Base Model is built below using the HoldOut Validation (train test split) to evaluate the accuracy

# ================ 6.1. Split the data to train and test set ==================

# Spot Check Algorithms with Decision Tree [8] [9] [10] and Linear Discriminant algorithms [11] [12]

# Data is splited into training and test set using the feature and target dataframes
# to understand the model performance with 70% training set and 30% test set
X_train, X_test, Y_train, Y_test = train_test_split(dfbankfeatures, dfbanktarget, test_size=0.3, random_state=None)

print('Total records of Train dataset: ', len(X_train))
print('Total attributes of Train dataset: ', len(X_train.columns))
print('Total records of Test dataset: ', len(X_test))

# Give a new line to clearly format the output
Newline()

# Output:
#Total records of Train dataset:  79627
#Total attributes of Train dataset:  7
#Total records of Test dataset:  34127

# Output after DATE indexed
#Total records of Train dataset:  81340
#Total attributes of Train dataset:  5
#Total records of Test dataset:  34861


# ========================== 6.2. Define models ===============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier [8] [9] [10]
# models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis [11] [12]
# models.append(('LDA', LinearDiscriminantAnalysis()))
# Create model object of Logistic Regression []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# Create model object of KNeighbors Classifier []
# models.append(('KNN', KNeighborsClassifier()))
# Create model object of Gaussian NB []
# models.append(('NB', GaussianNB()))
# Create model object of Support Vector Machine [] https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# models.append(('SVM', SVC(gamma='auto')))

# Print the time when the basic modeling starts
start_time = datetime.now()
print("Basic model start_time is - ", start_time)
# Give a new line to clearly format the output
Newline()

# Calling CreateModelObject function that creates model obejcts
CreateModelObject()

# ===================== 6.3. Build Model and Evaluate it  =====================

# Calling BasicModel function to build the basic models
BasicModel(models, X_train, X_test, Y_train, Y_test)

# Create a dataframe to store accuracy
dfbasicscore = pd.DataFrame(basic_score)    
print(dfbasicscore)

# Give a new line to clearly format the output
Newline()

# Print the time when the basic modeling ends
end_time = datetime.now()
print("Basic model end_time is - ", end_time)
# Print the total time spend to run the basic model
totaltime_basic = end_time - start_time
print("Total time to run the basic model is - ", totaltime_basic)
# Give a new line to clearly format the output
Newline()

# Output:
#  Model Name  Mean squared error  Root mean squared error  Coefficient of determination  
#0        DTR        9.131545e+02             3.021845e+01                  6.308531e-01
#1         LR        2.337413e+03             4.834680e+01                  5.509015e-02
#2        RFR        1.967358e+03             4.435491e+01                  2.046866e-01
#3         RR        2.078916e+03             4.559513e+01                  1.595888e-01
#4        LsR        2.081753e+03             4.562623e+01                  1.584418e-01
#5       SGDR        2.673263e+89             5.170361e+44                 -1.080679e+86
#6        KNR        1.234203e+03             3.513122e+01                  5.010679e-01
#7       LSVR        8.146268e+03             9.025668e+01                 -2.293166e+00
#8        SVR        2.474666e+03             4.974601e+01                 -3.949183e-04

# This is without any parameter in model object
#  Model Name  Mean squared error  Root mean squared error  Coefficient of determination
#0        DTR        9.131545e+02             3.021845e+01                   6.308531e-01
#1         LR        2.337413e+03             4.834680e+01                   5.509015e-02
#2        RFR        5.334698e+02             2.309696e+01                   7.843424e-01
#3       SGDR        1.331523e+89             3.649003e+44                  -5.382741e+85
#4        KNR        1.286433e+03             3.586689e+01                   4.799534e-01
#5       LSVR        8.146268e+03             9.025668e+01                  -2.293166e+00
#6        SVR        2.474650e+03             4.974586e+01                  -3.886260e-04

# Output after DATE Indexed
#Model Name  Mean squared error  Root mean squared error     Coefficient of determination
#0        DTR        6.472562e+03             8.045224e+01                   4.448717e-01
#1         LR        9.930738e+03             9.965309e+01                   1.482766e-01
#2        RFR        4.039285e+03             6.355537e+01                   6.535652e-01
#3       SGDR        6.034960e+58             2.456615e+29                  -5.175967e+54
#4        KNR        4.535254e+03             6.734429e+01                   6.110277e-01
#5       LSVR        1.735934e+14             1.317549e+07                  -1.488848e+10
#6        SVR        1.223494e+04             1.106117e+02                  -4.934691e-02

# The output shows that the accuracy for LDA model is better than Decision Tree Classifier.
# To improve the performance of each model, the evaluation is done with respect to different random_state in the next step

"""
################## STEP 7: BUILD MODELS WITH RANDOM STATE #####################

# Model is build with random state to evelaute the accuracy

# ======================= 7.1. Define models ==================================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
#models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
#models.append(('LDA', LinearDiscriminantAnalysis()))

# Calling CreateModelObject function that creates model obejcts
CreateModelObject()

# ===================== 7.2. Build Model and Evaluate it  =====================

# Define BuildModelRS function to evaluate the models based on the random states [8-12]
# rand_state variable has been defined in the config.py file with values 1,3,5,7
# stores the Model Name, Random State and Accuracy of each model in score list
def BuildModelRS(models,rand_state):
    # Evaluate each model in turn
    for name, model in models:
        # for loop will train and predict the decision tree model on different random states
        for n in rand_state:
            # The training set and test set has been splited using the feature and target dataframes with different random_state
            X_train, X_test, Y_train, Y_test = train_test_split(dfcreditdata, dfcredittarget, test_size=0.3, random_state=n)
            # Train Decision Tree Classifer
            modelfit = model.fit(X_train,Y_train)
            # Predict the response for test dataset
            Y_predict = modelfit.predict(X_test)
            # Store the accuracy in results
            results.append(metrics.accuracy_score(Y_test, Y_predict))
            # Store the model name in names
            names.append(name)
            # Store the Model Name, Random State and Accuracy into score list
            score.append({"Model Name": name, "Random State": int(n), "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
    return score

# Calling BuildModelRS to evaluate the models with random states and return the accuracy
BuildModelRS(models,rand_state)

# Create a dataframe to store accuracy
dfrsscore = pd.DataFrame(score)    
print(dfrsscore.head(8))

# Give a new line to clearly format the output
Newline()

# The output shows that the accuracy changes in each randon state. Therefore, data has been balanced in further steps.

######################## STEP 8: BALANCING THE DATA ###########################

# Installed imbalance-learn library using "conda install -c conda-forge imbalanced-learn" [15]

# To balance the data, RandomUnderSampler and SMOTE is utilized to undersample and oversample the data along with pipeline [16]

# Define oversmaple with SMOTE function
oversample = SMOTE()
# Define undersample with RandomUnderSampler function
undersample = RandomUnderSampler()
# Define Steps for oversample and undersample
steps = [('o', oversample), ('u', undersample)]
# Define the pipeline with the steps
pipeline = Pipeline(steps = steps)
# Fit the features and target using pipeline and resample them to get X and Y
X, Y = pipeline.fit_resample(dfcreditdata, dfcredittarget)
# Print the shape of X and Y
print("The features dataset has Rows {} and Columns {} ".format(X.shape[0], X.shape[1]))
print("The target dataset has Rows {} and Columns {} ".format(Y.shape[0],0))

# Give a new line to clearly format the output
Newline()

# By balancing the data using oversample and undersample, X and Y now have adequate amount of data avaialble
# Reducing the dimensionality is not needed because the dataset is small with only 15 features


######## STEP 9: OPTIMIZE MODEL ON BALANCED DATA USING CROSS VALIDATION #######

# StratifiedKFold Cross Validation is utilized to improve the performance as it splits the data approximately in the same percentage [17] [18]
# StratifiedKFold Cross Validation is used because there are 2 classes, and the split of train and test set could be properly done

# ============================ 9.1. Define models =============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))

# ===================== 9.2. Build Model and Evaluate it  =====================

# Define function BuildModelBalCV to evaluate models on the balanced data and utilize cross validation
def BuildModelBalCV(models):
    # Evaluate each model in turn  
    for name, model in models:
        # Define StratifiedKFold [17] [18]
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        # Store the accuracy in results
        results.append(cv_results)
        # Store the model name in names
        names.append(name)
        # Print the results
        print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean()*100, cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean()*100, "STD": cv_results.std()})
    return score
    return results
    return names

# Calling BuildModelBalCV function to get the accuracy, cross validation results and names of models used
BuildModelBalCV(models)
# Create a dataframe to store accuracy
dfscore = pd.DataFrame(score)    
print(dfscore.head())

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Outout:
    # On DTCLS: Mean is 82.355243 and STD is 0.033680
    # On LDA: Mean is 87.255477 and STD is 0.040058

# LDA is having the highest Mean value of 87.25%.
# Decision Tree has also performed good with Mean as 82.35%
# Looking at the boxplot and the Mean values, it is clear that LDA has performed better even after balancing the data 
# along with StratifiedKFold Cross Validation by giving Mean as 87.25%


################## STEP 10: TUNE DECISION TREE BY GRIDSEARCHCV ##################

# Tuning the Decision Tree Classifier with GridSearchCV to find the best hyperparameter [19] [20]

# Define decision tree classifier model
dtcmodel = DecisionTreeClassifier()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
param_dict = dict()
param_dict = {"criterion": ["gini", "entropy"], "max_depth": [7,5,3]}
# Build GridSearchCV to get the accuracy
search = GridSearchCV(dtcmodel, param_dict, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize 
print('Decision Tree Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Give a new line to clearly format the output
Newline()

# Output:
    # Decision Tree Mean Accuracy: 0.868290
    # Config: {'criterion': 'entropy', 'max_depth': 3}
# The best parameter for Decision Tree Classifier are 'criterion' as 'entropy', 'max_depth' as 3, 
# using which the performance of decision tree has been increased slightly with 86.82%

###################### STEP 11: TUNE LDA BY GRIDSEARCHCV ######################

# Tuning the LDA with GridSearchCV to find the best hyperparameter [19]

# Define LDA model
ldamodel = LinearDiscriminantAnalysis()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# Build GridSearchCV to get the accuracy
search = GridSearchCV(ldamodel, grid, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize
print('LDA Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Give a new line to clearly format the output
Newline()

# Output:
    # LDA Mean Accuracy: 0.872516
    # Config: {'solver': 'svd'}

# The best parameter LDA are 'solver' as 'svd' 
# using which the performance of LDA model has been increased slightly to 87.25%


######################### STEP 12: BUILD TUNED MODEL ##########################

# Building the model finally on balanced data with best hyper-parameters along with StratifiedKFold cross validation [21]

# ========================== 12.1. Define models ==============================

# Clear lists
ClearLists()

# Define models with the best hyperparameters found earlier
models.append(('DTCLS', DecisionTreeClassifier(criterion='entropy', max_depth=3)))
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))

# ==================== 12.2. Build Model and Evaluate it  =====================

# Define BuildFinalModel function to evaluate models with their hyper-parameters
def BuildFinalModel(models):
    # Evaluate each model in turn 
    for name, model in models:
        # Define StratifiedKFold
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation [21]
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        # Store the cross validationscore into results
        final_results.append(cv_results)
        # Store model name into names
        names.append(name)
        # Print the results
        print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean(), cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean(), "STD": cv_results.std()})
    return score
    return final_results

# Calling BuildFinalModel to get the accuracy after tuning
BuildFinalModel(models)
# Create a dataframe to store accuracy
dffinalscore = pd.DataFrame(score) 
print(dffinalscore.head())

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(final_results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Output:
    # On DTCLS: Mean is 0.859996 and STD is 0.034825
    # On LDA: Mean is 0.871264 and STD is 0.043806
    
# Looking at the Mean and Boxplot, it is clear that LDA has performed better than 
# Decision Tree after tuning the model with hyperparameters using StratifiedKFold cross validation. 
# However, there is not much difference in both the accuracies.


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     BUILDING DEEP LEARNING MODEL     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Neural network is considered for implementing the deep learning model below [22] [23]

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

####################### STEP 2: BUILD BASE MODEL ##############################

# ========================== 2.1. Define models ===============================

# Clear list
ClearLists()

# Define Sequential model
model = Sequential()
# Define the Flatten layer
model.add(Flatten())
# Define the hidden layer
model.add(Dense(128, input_dim=15, activation='relu'))
# Define the output layer
model.add(Dense(2, activation='softmax'))

# ==================== 2.2. Build Model and Evaluate it  ======================

# Define BasicDeepLearnModel function to evaluate the model and get the accuracy

def BasicDeepLearnModel(model):
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    # Fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=100)
    # Evaluate the model
    scores = model.evaluate(x_test, y_test)
    # Print the accuracy
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # Store the Metrics Name and Accuracy into scores_list list
    scores_list.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})
    return scores_list

# Calling BasicDeepLearnModel function to evaluate the deep learning model without any hyper-parameters
BasicDeepLearnModel(model)

# Create a dataframe to store accuracy
dfBasickeras = pd.DataFrame()
# Define a dataframe to with scores_list
dfBasickeras = pd.DataFrame(scores_list) 
print(dfBasickeras.head())

# Give a new line to clearly format the output
Newline()

# Output: 
    # accuracy: 79.53%
# The Accuracy of Neural Nterwork deep learning model is 79.534882. Observed that loss is between 0.3 to 3.9.


################# STEP 3: BUILD MODEL WITH VARIOUS EPOCHS #####################

# ========================== 3.1. Define models ===============================

# Clear list
ClearLists()

# Define Sequential model
model = Sequential()
# Define the Flatten layer
model.add(Flatten())
# Define the hidden layer
model.add(Dense(128, input_dim=15, activation='relu'))
# Define the output layer
model.add(Dense(2, activation='softmax'))

# ==================== 3.2. Build Model and Evaluate it  ======================

# Define BasicDeepLearnModel function to evaluate the model and get the accuracy

def BasicDeepLearnModel(model):
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    for n in eps:
        model.fit(x_train, y_train, epochs=n, batch_size=100)
        # Evaluate the model
        scores = model.evaluate(x_test, y_test)
        # Print the accuracy
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # Store the Metrics Name and Accuracy into scores_list list
        scores_list.append({"Epochs": n, "Accuracy": scores[1]*100})
    return scores_list

# Calling BasicDeepLearnModel function to evaluate the deep learning model without any hyper-parameters
BasicDeepLearnModel(model)

# Create a dataframe to store accuracy
dfepochskeras = pd.DataFrame()
# Define a dataframe to with scores_list
dfepochskeras = pd.DataFrame(scores_list) 
print(dfepochskeras.head())

# Give a new line to clearly format the output
Newline()

# Find the maximum accuracy found with various epochs
df1 = dfepochskeras[["Accuracy"]].idxmax()
ind = df1.values[0]
df2 = dfepochskeras.iloc[ind:ind+1,:1]
max_epoch = df2["Epochs"].values[0]
print("The best epochs value having maximum accuracy is: ")
print (max_epoch)

# The Accuracy with the best epoch value of Neural Nterwork deep learning model is 83.720928


################# STEP 4: TUNE MODEL WITH HYPERPARAMETERS #####################

# To tune the Deep Learning Model, kerastuner has been installed using "conda install -c conda-forge keras-tuner" [24]
# kerastuner was throwing error even after installing it with above command, so installed it with "pip install keras-tuner" [25]

# ================== 4.1. Find best hyper-parameters ==========================

# Define function DLbuildmodel with the hyperparameters [26]
def DLbuildmodel(hp):
    # Define Sequential model
    model = Sequential()
    # To get the best value of number of neurons per hidden layer, hp.Int hyperparameter is used 
    # by giving a range from min_value as 32 and max_value as 300 with the step size as 50
    model.add(Dense(units= hp.Int('units', min_value=32, max_value=300, step=50), activation='relu'))
    # Define the output layer
    model.add(Dense(2, activation='softmax'))
    # compile model - To get the best Adam optimizer, learning_rate from 0.01 and 0.001 is used
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-4])),metrics=['accuracy'])
    return model
    
# ================ 4.2. Tune model with RandomSearch ==========================

# To create an instance of RandomSearch class, RandomSearch function is called
# Calling the model building funtion "DLbuildmodel"
# To maximize the accuracy "objective" is set to "val_accuracy"
# max_trails is set to 5 to limit the number of model variations to test
# executions_per_trial is set to 3 to limit the number of trials per variation

tuner = RandomSearch(DLbuildmodel, objective = 'val_accuracy', max_trials=5, executions_per_trial=3, directory = 'my_dir', project_name='deeplearningmodel')

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
# Train the model
tuner.search(x_train,y_train,epochs=max_epoch,validation_data=(x_test,y_test))
# Retrieve the best_model
best_model = tuner.get_best_models(num_models=1)[0]
# Get the loss and mse from the best model
loss, mse = best_model.evaluate(x_test, y_test)

# Give a new line to clearly format the output
Newline()

# Summarize the results with best hyper-parameters
tuner.search_space_summary()

# Give a new line to clearly format the output
Newline()

# Print the best hyperparameter
print(tuner.get_best_hyperparameters()[0].values)

# Give a new line to clearly format the output
Newline()

# Output:
    # Best val_accuracy So Far: 0.8790697654088339
    # Total elapsed time: 00h 06m 25s
    # INFO:tensorflow:Oracle triggered exit
    # 7/7 [==============================] - 0s 1ms/step - loss: 0.4322 - accuracy: 0.8599


    # Search space summary
    # Default search space size: 2
    # units (Int)
    # {'default': None, 'conditions': [], 'min_value': 32, 'max_value': 300, 'step': 50, 'sampling': None}
    # learning_rate (Choice)
    # {'default': 0.01, 'conditions': [], 'values': [0.01, 0.0001], 'ordered': True}


    # {'units': 132, 'learning_rate': 0.01}

####################### STEP 5: Final Deep Learning Model #####################

# Define list
scoreslist = []
# Clear list
scoreslist.clear()

# Define model
model = Sequential()
# Define the layers with best units value
model.add(Dense(132, input_dim=15, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile model with best learning_rate
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Split the dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)

# fit the model with best epoch value
modelfit = model.fit(x_train, y_train, epochs=max_epoch, batch_size=100)

# evaluate the model
scores = model.evaluate(x_test, y_test)

# print the accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scoreslist.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})

# Create a dataframe to store accuracy
dfkeras = pd.DataFrame()
#scoreslist.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})
dfkeras = pd.DataFrame(scoreslist) 
print(dfkeras.head())

# Output:
    # Observed that the accuracy varies between 80% to 94% after applying the best epoch value and best hyperparameter 
    # It is seen that the last trained model with epoch as 300 gives accuracy as 93.46%, however the best accuracy is 86.05%
    # Observed that loss is between 0.19 to 0.25


#################### STEP 6: PLOT LOSS AND ACCURACY ###########################

# Plotting Loss and Accuracy [27]

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(modelfit.history['loss'], label='X_train')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(modelfit.history['accuracy'], label='X_test')
pyplot.legend()
pyplot.show()

# By the loss and accuracy plotted below, it is observed that the accuracy is increased by the decrease in loss.

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     CONCLUSION OF MODELS BUILT     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Decision Tree model produced an accuracy of 81.63% on single modelling training
# Linear Discriminant Classifier model produced an accuracy of 85.20% on single modelling training
# The Decision Tree and Linear Discriminant Classifier models performed good after balancing the data and 
# applying few hyperparameter along with cross validation method. 
# Performance of Decision Tree model increased to 85.99%, which is an increase of 4.36% after tuning the model
# whereas LDA model performed slightly better than Decision Tree by giving accuracy of 87.12%,
# which is an increase of just 1.92% after tuning the model.
# There wasn't much difference between both models after they were tuned.
# The Neural Network Deep Learning model produced an accuracy of 79.53% on single modelling training. 
# The performance of Neural Network is good with 85.05% after tuning the model with hyperparameters, 
# which is a significant increase of 6.52% after the model was tuned.
# Also, the loss has been drastically changed from (0.3 to 3.9) to (0.19 to 0.25).
# Considering only the accuracy of the three model, LDA has the highest accuracy. 
# however Neural Network performed much better than Decision Tree and LDA models after it was tuned.
# Hence Neural Network Deep Learning model is recommended to be utilized to prove the approval of credit cards.


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ REFERNCES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# [1] https://archive.ics.uci.edu/ml/datasets/Credit+Approval
# [4] https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
# [5] https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder
# [6] https://pythonbasics.org/seaborn-pairplot/
# [7] https://seaborn.pydata.org/generated/seaborn.heatmap.html
# [8] https://scikit-learn.org/stable/modules/tree.html
# [9] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# [10] https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# [11] https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
# [12] https://www.statology.org/linear-discriminant-analysis-in-python/
# [13] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# [14] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# [15] https://anaconda.org/conda-forge/imbalanced-learn
# [16] https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# [17] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/
# [18] https://scikit-learn.org/stable/modules/cross_validation.html
# [19] https://machinelearninghd.com/gridsearchcv-classification-hyper-parameter-tuning/
# [20] https://ai.plainenglish.io/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
# [21] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/
# [22] https://keras.io/guides/training_with_built_in_methods/
# [23] https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# [24] https://anaconda.org/conda-forge/keras-tuner
# [25] https://keras.io/api/keras_tuner/tuners/
# [26] https://keras.io/api/keras_tuner/hyperparameters/#hyperparameters-class
# [27] https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/


"""