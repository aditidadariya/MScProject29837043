#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibPreProcess_2.py

This file contains all the functions to be called Pre-processing of dataset.

"""
######################## Importing necessary libraries ########################
import pandas as pd
# Import LabelEncoder to change the data types of attributes
from sklearn.preprocessing import LabelEncoder
# Import StandardScaler to Standardize the data
from sklearn.preprocessing import StandardScaler
# Import Config.py file to get all the variables here
from Config_1 import *
#import os
import requests
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import time

########################## Declarations of Pre-Processing Functions ##############################

# Newline gives a new line
def Newline():
    print("\r\n")
    
# ClearLists removes all values from lists
def ClearLists():
    models.clear()
    names.clear()
    results.clear()
    basic_score.clear()
    score.clear()
    final_results.clear()
    scores_list.clear()
    
 
# ReadDataset function reads the dataset csv file and store it in a dataframe
# location_of_file is variable thats stores the file location and is defined in config.py file
# https://www.kaggle.com/datasets/apoorvwatsky/bank-transaction-data/code?select=bank.xlsx
def ReadDataset(location_of_file, filename):
    # Read the data file from specified location
    df = pd.read_excel(location_of_file + filename)
    # Read the data file from specified location and make DATE column as index
    #df = pd.read_excel(location_of_file, index_col="DATE")
    # Replace space from the column names in the dataframe
    df.columns = df.columns.str.replace(' ','_')
    # Translate all the column names to uppercase
    df.columns = [x.upper() for x in df.columns]
    # Adding ms value along with date to handle the duplicate date
    df['DATE'] = df['DATE'] + pd.to_timedelta(df.groupby('DATE').cumcount(), unit='ms')
    # Setting DATE as index
    df.set_index('DATE', inplace = True)
    # Sorting the index
    df = df.sort_index()
    return df


# FillEmptyColumn function is defined to update the empty columns with some value
def FillEmptyColumn(df,columnname,value):
    while df[columnname].isnull().sum() > 0:
        df[columnname] = df[columnname].replace(np.nan,value)
  
# RemoveSpecialCharacter function defined to remove the special character "'" from ACCOUNT_NO attribute
def RemoveSpecialCharacter(df,replacecolumn, specialchar):
    #dfbankdata["ACCOUNT_NO"].replace(to_replace = "'",value='',inplace = True,regex=True)
    df[replacecolumn].replace(to_replace = specialchar,value='',inplace = True,regex=True)

# ReplaceCategory fucton is defined to update the TRANSACTION_DETAILS values to lower case to maintain consistancy [] https://stackoverflow.com/questions/39768547/replace-whole-string-if-it-contains-substring-in-pandas
def ReplaceCategory(df,featurename,categorylist):
    for ele in categorylist:
        elelower = ele.lower()
        eleupper = ele.upper()
        #if df[featurename].isnull().sum() == 0:
        df.loc[df[featurename].str.contains(elelower, case=True), featurename] = ele
        df.loc[df[featurename].str.contains(eleupper, case=True), featurename] = ele
        df.loc[df[featurename].str.contains(ele, case=True), featurename] = ele
        #else:
           # print("There are null values")
    return df

# Replace the Reference Numbers in TRANSACTION_DETAILS column to RefNum 
# [] https://stackoverflow.com/questions/68255137/replace-value-in-a-dataframe-column-if-value-starts-with-a-specific-character
def ReplaceIntegerCategory(df,replacecolumn):
    df[replacecolumn]=np.where([i.startswith("0")|i.startswith("1")|i.startswith("2")|i.startswith("3")
                                        |i.startswith("4")|i.startswith("5")|i.startswith("6")|i.startswith("7")
                                        |i.startswith("8")|i.startswith("9") for i in df[replacecolumn]],'RefNum',df[replacecolumn])
    df.to_csv('banklist.csv')
    return df

# CheckValueExists function is defined to check if the value exists in a list
# [] https://stackoverflow.com/questions/7571635/fastest-way-to-check-if-a-value-exists-in-a-list
def CheckValueExists(element, collection: iter):
    return element in collection

# RenameAllCategory function is defined to reduce the number of categories by renaming it the common values
def RenameAllCategory(df,replacecolumn):
    for ind, val in enumerate(df[replacecolumn]):
        if CheckValueExists(val, businesslist) == True:
            df.replace(to_replace = val,inplace = True,value='Business')
        elif CheckValueExists(val, courierservicelist) == True:
            df.replace(to_replace = val,inplace = True,value='CourierService')
        elif CheckValueExists(val, investmentlist) == True:
            df.replace(to_replace = val,inplace = True,value='Investment')
        elif CheckValueExists(val, mobileservicelist) == True:
            df.replace(to_replace = val,inplace = True,value='MobileService')
        elif CheckValueExists(val, transfermodelist) == True:
            df.replace(to_replace = val,inplace = True,value='TransferMode')
        elif CheckValueExists(val, chargeslist) == True:
            df.replace(to_replace = val,inplace = True,value='Charges')
        elif CheckValueExists(val, entertaintravellist) == True:
            df.replace(to_replace = val,inplace = True,value='EntertainmentTravel')
        elif CheckValueExists(val, taxlist) == True:
            df.replace(to_replace = val,inplace = True,value='Tax')
    return df


# ReplaceIndividualTrans function is defined to replace the Individual transactions
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
def ReplaceIndividualTrans(df,replacecolumn,catlist):
    for ind, val in enumerate(df[replacecolumn]):
        if CheckValueExists(val, catlist) == False:
            df.replace(to_replace = val,inplace = True,value='Individual')
    df = RenameAllCategory(df,replacecolumn) 
    df.to_csv('banklist.csv')
    return df
    
# FindOutliers_IQR function is defined to find the outliers in Withdrawal_AMT and Deposit_AMT attributes
# https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/
def FindOutliers_IQR(df):
    # Get the 25% quantile
    q1=df.quantile(0.25)
    print("25% quantile is " + str(q1))
    # Get the 75% quantile
    q3=df.quantile(0.75)
    print("75% quantile is " + str(q3))
    # Get IQR
    IQR=q3-q1
    print("IQR is " + str(IQR))
    # Get the outliers
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    # Get the upper limit
    upper = df[~(df>(q3+1.5*IQR))].max()
    # Get the lower limit
    lower = df[~(df<(q1-1.5*IQR))].min()
    print('number of outliers: '+ str(len(outliers)))
    #print('max outlier value: '+ str(outliers.max()))
    #print('min outlier value: '+ str(outliers.min()))
    # Replace the outlier using imputation as is they were missing values
    df = np.where(df > upper, df.mean(), np.where(df < lower,df.mean(),df))
    
    # Give a new line to clearly format the output
    Newline()
    
    return outliers, df

# StdData function is to standardize the dataset []
# [] https://stackoverflow.com/questions/49641707/standardize-some-columns-in-python-pandas-dataframe
# [] https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
def StdData(df):
    # define standard scaler
    scaler = StandardScaler()
    # scaler will be used in the model prediction and hence fit and transform is done separately
    #dftoscale = df[cols]
    #print(dftoscale)
    scaler = scaler.fit(df)
    df = scaler.transform(df)
    # transform dataset as this will be used in the regresion models and visualization
    #stdscaler = StandardScaler()
    #df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'TRANSACTION_AMOUNT', 'BALANCE_AMT']] = scaler.fit_transform(df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'TRANSACTION_AMOUNT', 'BALANCE_AMT']])
    #df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT', 'TRANSACTION_AMOUNT']] = stdscaler.fit_transform(df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT', 'TRANSACTION_AMOUNT']])
    return df, scaler
    #return df

# InvTransDF function is defined to inverse transform the data
def InvTransDF(df, scaler):
    df_array = df.to_numpy()
    #print(df_test_array)
    finalresult = scaler.inverse_transform(df_array)
    #finalresult
    df_finalresult = pd.DataFrame(finalresult, columns = ['TRANSACTION_AMOUNT','Predictions'])
    return df_finalresult
 
# Encoder function transforms the character and object value in dataframe [5]
def Encoder(df, cols):     
    # cols variable is to store the column names where encoding is needed [1]     
    #cols = ['TRANSACTION_DETAILS']
    # Encode labels of multiple columns at once
    df[cols] = df[cols].apply(LabelEncoder().fit_transform)
    return df


    
# References:
# [1] https://vitalflux.com/labelencoder-example-single-multiple-columns/
# [6] https://pythonbasics.org/seaborn-pairplot/
# [7] https://seaborn.pydata.org/generated/seaborn.heatmap.html

