#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibPreProcess.py

This file contains all the functions to be called by mainsfile.py

"""
######################## Importing necessary libraries ########################
import pandas as pd
# Import LabelEncoder to change the data types of attributes
from sklearn.preprocessing import LabelEncoder
# Import Config.py file to get all the variables here
from Config import *
#import os
import requests
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import time

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
# Import mean_squared_error and r2_score
#from sklearn.metrics import mean_squared_error, r2_score
# Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics
# Import StandardScaler to Standardize the data
from sklearn.preprocessing import StandardScaler


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
def ReadDataset(location_of_file):
    # Read the data file from specified location
    #df = pd.read_excel(location_of_file)
    # Read the data file from specified location and make DATE column as index
    df = pd.read_excel(location_of_file, index_col="DATE")
    # Replace space from the column names in the dataframe
    df.columns = df.columns.str.replace(' ','_')
    # Translate all the column names to uppercase
    df.columns = [x.upper() for x in df.columns]
    return df

"""
# for loop will iterate over all the columns in dataframe to find and drop the rows with special characters [4]
def RemoveSpecialChar(special_char,df):
    for eachcol in df:
        # Drop rows where "?" is displayed
        df.drop(df.loc[df[eachcol] == special_char].index, inplace=True)
    return df

"""

# FillEmptyColumn function is defined to update the empty columns with some value
def FillEmptyColumn(df,columnname,value):
    while df[columnname].isnull().sum() > 0:
        df[columnname] = df[columnname].replace(np.nan,value)
  
# RemoveSpecialCharacter function defined to remove the special character "'" from ACCOUNT_NO attribute
def RemoveSpecialCharacter(df,replacecolumn, specialchar):
    #dfbankdata["ACCOUNT_NO"].replace(to_replace = "'",value='',inplace = True,regex=True)
    df[replacecolumn].replace(to_replace = specialchar,value='',inplace = True,regex=True)

# ConvertDataTypeStr function is defined to convert datatype of TRANSACTION_DETAILS attribute to string
def ConvertDataTypeStr(df,columnname):
    df = df.astype({columnname: str})
    return df

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



"""
# DropRows function is defined to drop the unwanted rows of TRANSACTION_DETAILS attirbute
def DropRows(df):
    # Get indexes where TRANSACTION_DETAILS column doesn't have values of categorylist
    #indexNames = dfbankdataset[~((dfbankdataset['TRANSACTION_DETAILS'] == "Indiaforensic") 
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "fdrl")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "npci")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "dsb")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "sbi")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "oxygen")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "hdfc") 
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "neft") 
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "imps")
    #                  | (dfbankdataset['TRANSACTION_DETAILS'] == "NoTransDetail"))].index 
    indexNames =  df[~((df['TRANSACTION_DETAILS'] == '2DFINE') | (df['TRANSACTION_DETAILS'] == 'AAB ENTERPRISES') | (df['TRANSACTION_DETAILS'] == 'ACCOUNT') | (df['TRANSACTION_DETAILS'] == 'ACQ') 
                       | (df['TRANSACTION_DETAILS'] == 'ADJ') | (df['TRANSACTION_DETAILS'] ==  'AEPS') | (df['TRANSACTION_DETAILS'] == 'AIRPAY') | (df['TRANSACTION_DETAILS'] == 'AIRTEL') 
                       | (df['TRANSACTION_DETAILS'] == 'ALT DIGITAL') | (df['TRANSACTION_DETAILS'] == 'APPNIT') | (df['TRANSACTION_DETAILS'] == 'ATLAS') | (df['TRANSACTION_DETAILS'] == 'ATM') 
                       | (df['TRANSACTION_DETAILS'] == 'AVENUES INDIA PRIVATE LIM') | (df['TRANSACTION_DETAILS'] == 'AWARDS') | (df['TRANSACTION_DETAILS'] == 'AXEL') | (df['TRANSACTION_DETAILS'] == 'BAJAJ ALLIANZ') 
                       | (df['TRANSACTION_DETAILS'] == 'Bank Guar') | (df['TRANSACTION_DETAILS'] == 'BASIST') | (df['TRANSACTION_DETAILS'] == 'BBPS') | (df['TRANSACTION_DETAILS'] == 'BEAT CSH') 
                       | (df['TRANSACTION_DETAILS'] == 'BEST') | (df['TRANSACTION_DETAILS'] == 'BHAGALPUR ELECTRICITY DIS') | (df['TRANSACTION_DETAILS'] == 'BIDDERBOY CORPORATION') | (df['TRANSACTION_DETAILS'] == 'BIGTREE ENTERTAINMENT PRI') 
                       | (df['TRANSACTION_DETAILS'] == 'BILLPAY') | (df['TRANSACTION_DETAILS'] == 'BIRLA') | (df['TRANSACTION_DETAILS'] == 'BLUE DART') | (df['TRANSACTION_DETAILS'] == 'BONUSHUB') 
                       | (df['TRANSACTION_DETAILS'] == 'BOOK') | (df['TRANSACTION_DETAILS'] == 'BOWMAN') | (df['TRANSACTION_DETAILS'] == 'BSES RAJDHANI POWER LIMIT') | (df['TRANSACTION_DETAILS'] == 'BSES YAMUNA POWER LIMITED') 
                       | (df['TRANSACTION_DETAILS'] == 'CALLHEALTH ONLINE SERVICE') | (df['TRANSACTION_DETAILS'] == 'CASH') | (df['TRANSACTION_DETAILS'] == 'CATENA TECHNOLOGIES PRIVA') | (df['TRANSACTION_DETAILS'] == 'CB')
                       | (df['TRANSACTION_DETAILS'] == 'CDUTWL') | (df['TRANSACTION_DETAILS'] == 'CESS') | (df['TRANSACTION_DETAILS'] == 'Charges') | (df['TRANSACTION_DETAILS'] == 'CHQ') 
                       | (df['TRANSACTION_DETAILS'] == 'CLEARCAR RENTAL PRIVATE L') | (df['TRANSACTION_DETAILS'] == 'Closure') | (df['TRANSACTION_DETAILS'] == 'COMMDEL CONSUTING SERVICE') | (df['TRANSACTION_DETAILS'] == 'CONFEDERATION OF INDIAN I')
                       | (df['TRANSACTION_DETAILS'] == 'CREDIT') | (df['TRANSACTION_DETAILS'] == 'CRY') | (df['TRANSACTION_DETAILS'] == 'CULLIGENCE SOFTWARE') | (df['TRANSACTION_DETAILS'] == 'DD') 
                       | (df['TRANSACTION_DETAILS'] == 'DEFERRED') | (df['TRANSACTION_DETAILS'] == 'DEN NETWORKS LIMITED') | (df['TRANSACTION_DETAILS'] == 'DHL') | (df['TRANSACTION_DETAILS'] == 'DIGITAL') 
                       | (df['TRANSACTION_DETAILS'] == 'DISH INFRA SERVICES PRIVA') | (df['TRANSACTION_DETAILS'] == 'Dr.') | (df['TRANSACTION_DETAILS'] == 'Draw Down') | (df['TRANSACTION_DETAILS'] == 'dsb') 
                       | (df['TRANSACTION_DETAILS'] == 'E SUVIDHA') | (df['TRANSACTION_DETAILS'] == 'E-BILLING SOLUTIONS PRIVA') | (df['TRANSACTION_DETAILS'] == 'EIH') | (df['TRANSACTION_DETAILS'] == 'ELEGANCE FASHION AND YOU') 
                       | (df['TRANSACTION_DETAILS'] == 'EMBEE SOFTWARE PVT LTD') | (df['TRANSACTION_DETAILS'] == 'ETRAVELVALUE') | (df['TRANSACTION_DETAILS'] == 'FACEBOOK') | (df['TRANSACTION_DETAILS'] == 'FAKE') 
                       | (df['TRANSACTION_DETAILS'] == 'FD') | (df['TRANSACTION_DETAILS'] == 'fdrl') | (df['TRANSACTION_DETAILS'] == 'FEDEX') | (df['TRANSACTION_DETAILS'] == 'FEE') 
                       | (df['TRANSACTION_DETAILS'] == 'FERNS') | (df['TRANSACTION_DETAILS'] == 'FINSPIRE SOLUTIONS PRIVAT') | (df['TRANSACTION_DETAILS'] == 'FIXED MOBILE') | (df['TRANSACTION_DETAILS'] == 'FROM RBL BANK LTD') 
                       | (df['TRANSACTION_DETAILS'] == 'FT ZL') | (df['TRANSACTION_DETAILS'] == 'FTR') | (df['TRANSACTION_DETAILS'] == 'FUJIAN') | (df['TRANSACTION_DETAILS'] == 'FUTURE RETAIL LIMITED') 
                       | (df['TRANSACTION_DETAILS'] == 'FX CONV CHGS') | (df['TRANSACTION_DETAILS'] == 'GOLD') | (df['TRANSACTION_DETAILS'] == 'GOLF') | (df['TRANSACTION_DETAILS'] == 'GST') 
                       | (df['TRANSACTION_DETAILS'] == 'HAPPYDEAL18') | (df['TRANSACTION_DETAILS'] == 'hdfc') | (df['TRANSACTION_DETAILS'] == 'HINDUSTAN SERVICE STATION') | (df['TRANSACTION_DETAILS'] == 'HOSPITAL') 
                       | (df['TRANSACTION_DETAILS'] == 'HUDA') | (df['TRANSACTION_DETAILS'] == 'IFSC') | (df['TRANSACTION_DETAILS'] == 'IMAR') | (df['TRANSACTION_DETAILS'] == 'imps') 
                       | (df['TRANSACTION_DETAILS'] == 'Indfor') | (df['TRANSACTION_DETAILS'] == 'Indiaforensic') | (df['TRANSACTION_DETAILS'] == 'INDIAIDEAS') | (df['TRANSACTION_DETAILS'] == 'INDIAN OIL') 
                       | (df['TRANSACTION_DETAILS'] == 'INDO') | (df['TRANSACTION_DETAILS'] == 'INFOEDGE INDIA LTD') | (df['TRANSACTION_DETAILS'] == 'INT') | (df['TRANSACTION_DETAILS'] == 'INWARD') 
                       | (df['TRANSACTION_DETAILS'] == 'IO') | (df['TRANSACTION_DETAILS'] == 'IRCTC') | (df['TRANSACTION_DETAILS'] == 'IRTT') | (df['TRANSACTION_DETAILS'] == 'IW CLG') 
                       | (df['TRANSACTION_DETAILS'] == 'IWSTHCTS1') | (df['TRANSACTION_DETAILS'] == 'JANA') | (df['TRANSACTION_DETAILS'] == 'JUST DIAL LIMITED') | (df['TRANSACTION_DETAILS'] == 'KAYGEES') 
                       | (df['TRANSACTION_DETAILS'] == 'KRYPTOS MOBILE PRIVATE LI') | (df['TRANSACTION_DETAILS'] == 'LINKEDIN SINGAPORE PTE LT') | (df['TRANSACTION_DETAILS'] == 'LOCALCUBE COMMERCE PRIVAT') | (df['TRANSACTION_DETAILS'] == 'MAD') 
                       | (df['TRANSACTION_DETAILS'] == 'MAESTRO') | (df['TRANSACTION_DETAILS'] == 'MARGIN') | (df['TRANSACTION_DETAILS'] == 'MARKING') | (df['TRANSACTION_DETAILS'] == 'MASTER') 
                       | (df['TRANSACTION_DETAILS'] == 'MAW') | (df['TRANSACTION_DETAILS'] == 'MDR') | (df['TRANSACTION_DETAILS'] == 'MEHAR ENTERTAINMENT PRIVA') | (df['TRANSACTION_DETAILS'] == 'MEHER ENTERTAINMENT PRIVA') 
                       | (df['TRANSACTION_DETAILS'] == 'MERU CAB COMPANY PRIVATE') | (df['TRANSACTION_DETAILS'] == 'METRO INFRASYS PRIVATE LI') | (df['TRANSACTION_DETAILS'] == 'MICRO') | (df['TRANSACTION_DETAILS'] == 'MS EMBEE SOFTWARE') 
                       | (df['TRANSACTION_DETAILS'] == 'MTNL') | (df['TRANSACTION_DETAILS'] == 'MUZAFFARPUR VIDYUT VITARA') | (df['TRANSACTION_DETAILS'] == 'NATIONAL') | (df['TRANSACTION_DETAILS'] == 'neft') 
                       | (df['TRANSACTION_DETAILS'] == 'NETHOPE') | (df['TRANSACTION_DETAILS'] == 'NFS') | (df['TRANSACTION_DETAILS'] == 'NOMISMA MOBILE SOLUTIONS') | (df['TRANSACTION_DETAILS'] == 'NORTH DELHI POWER LTD') 
                       | (df['TRANSACTION_DETAILS'] == 'NoTransDetail') | (df['TRANSACTION_DETAILS'] == 'npc') | (df['TRANSACTION_DETAILS'] == 'OBEROI') | (df['TRANSACTION_DETAILS'] == 'OFT') 
                       | (df['TRANSACTION_DETAILS'] == 'OM GANESH') | (df['TRANSACTION_DETAILS'] == 'OMNICOM MEDIA GROUP INDIA') | (df['TRANSACTION_DETAILS'] == 'ONLINE') | (df['TRANSACTION_DETAILS'] == 'ORAVEL STAYS PRIVATE LIMI') 
                       | (df['TRANSACTION_DETAILS'] == 'ORTT') | (df['TRANSACTION_DETAILS'] == 'OUTWARD') | (df['TRANSACTION_DETAILS'] == 'OW RET') | (df['TRANSACTION_DETAILS'] == 'oxygen') 
                       | (df['TRANSACTION_DETAILS'] == 'PASFAR TECHNOLOGIES PRIVA') | (df['TRANSACTION_DETAILS'] == 'PAYGATE INDIA PRIVATE LIM') | (df['TRANSACTION_DETAILS'] == 'Payments') | (df['TRANSACTION_DETAILS'] == 'Payoff') 
                       | (df['TRANSACTION_DETAILS'] == 'PENALITY') | (df['TRANSACTION_DETAILS'] == 'PENALTY') | (df['TRANSACTION_DETAILS'] == 'PENDING') | (df['TRANSACTION_DETAILS'] == 'PENLTY') 
                       | (df['TRANSACTION_DETAILS'] == 'PVR LIMITED') | (df['TRANSACTION_DETAILS'] == 'QUINSEL SERVICES PRIVATE') | (df['TRANSACTION_DETAILS'] == 'QWIKCILVER SOLUTIONS') | (df['TRANSACTION_DETAILS'] == 'RAI') 
                       | (df['TRANSACTION_DETAILS'] == 'RAJMATAJI') | (df['TRANSACTION_DETAILS'] == 'RECOVERY') | (df['TRANSACTION_DETAILS'] == 'REJ') | (df['TRANSACTION_DETAILS'] == 'REP') 
                       | (df['TRANSACTION_DETAILS'] == 'Repayment') | (df['TRANSACTION_DETAILS'] == 'REV') | (df['TRANSACTION_DETAILS'] == 'REVERSE LOGISTICS COMPANY') | (df['TRANSACTION_DETAILS'] == 'RF') 
                       | (df['TRANSACTION_DETAILS'] == 'RMCPL') | (df['TRANSACTION_DETAILS'] == 'ROYAL INFOSYS') | (df['TRANSACTION_DETAILS'] == 'RPT') | (df['TRANSACTION_DETAILS'] == 'RTGS') 
                       | (df['TRANSACTION_DETAILS'] == 'RUPAY') | (df['TRANSACTION_DETAILS'] == 'sbi') | (df['TRANSACTION_DETAILS'] == 'SEND MONEY') | (df['TRANSACTION_DETAILS'] == 'SERVICE TX') 
                       | (df['TRANSACTION_DETAILS'] == 'SHENZHEN JUSTTIDE TECH CO') | (df['TRANSACTION_DETAILS'] == 'SHREEJEE PRIME INFOTECH P') | (df['TRANSACTION_DETAILS'] == 'SKORYDOV') | (df['TRANSACTION_DETAILS'] == 'SMARTSERV INFOSERVICES PR') 
                       | (df['TRANSACTION_DETAILS'] == 'SND LIMITED') | (df['TRANSACTION_DETAILS'] == 'SONATA FINANCE PRIVATE LI') | (df['TRANSACTION_DETAILS'] == 'SPORTS') | (df['TRANSACTION_DETAILS'] == 'SRS LIMITED')
                       | (df['TRANSACTION_DETAILS'] == 'SRVC TX') | (df['TRANSACTION_DETAILS'] == 'STAMP') | (df['TRANSACTION_DETAILS'] == 'SWACHH') | (df['TRANSACTION_DETAILS'] == 'SYNERGISTIC FINANCIAL NET') 
                       | (df['TRANSACTION_DETAILS'] == 'TAKEOVER') | (df['TRANSACTION_DETAILS'] == 'TATA') | (df['TRANSACTION_DETAILS'] == 'TAX') | (df['TRANSACTION_DETAILS'] == 'TDS') 
                       | (df['TRANSACTION_DETAILS'] == 'TELECOM') | (df['TRANSACTION_DETAILS'] == 'TELEPOWER') | (df['TRANSACTION_DETAILS'] == 'TELPOWER')| (df['TRANSACTION_DETAILS'] == 'THE UNIWORLD CITY')
                       | (df['TRANSACTION_DETAILS'] == 'TITAN') | (df['TRANSACTION_DETAILS'] == 'TNT') | (df['TRANSACTION_DETAILS'] == 'TO For') | (df['TRANSACTION_DETAILS'] == 'TPF')
                       | (df['TRANSACTION_DETAILS'] == 'TRANGLO') | (df['TRANSACTION_DETAILS'] == 'TransNum') | (df['TRANSACTION_DETAILS'] == 'TRAVEL') | (df['TRANSACTION_DETAILS'] == 'TRF') 
                       | (df['TRANSACTION_DETAILS'] == 'UPI') | (df['TRANSACTION_DETAILS'] == 'USD') | (df['TRANSACTION_DETAILS'] == 'VAISHANVI') | (df['TRANSACTION_DETAILS'] == 'VENDOR') 
                       | (df['TRANSACTION_DETAILS'] == 'VINILOK SOLUTIONS') | (df['TRANSACTION_DETAILS'] == 'VISA') | (df['TRANSACTION_DETAILS'] == 'VODAFONE') | (df['TRANSACTION_DETAILS'] == 'VOUCHER')
                       | (df['TRANSACTION_DETAILS'] == 'VOYLLA FASHIONS PRIVATE L') | (df['TRANSACTION_DETAILS'] == 'WAIVE') | (df['TRANSACTION_DETAILS'] == 'WORLD WIDE FUND') | (df['TRANSACTION_DETAILS'] == 'ZAAK EPAYMENT SERVICES PR') 
                       | (df['TRANSACTION_DETAILS'] == 'ZEN'))].index
    # Delete these row indexes from dataFrame
    #df.drop(indexNames , inplace=True)
    # Save the dataset in .csv file
    df.to_csv('banklist.csv')
    return df

"""

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
# #Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
def ReplaceIndividualTrans(df,replacecolumn,catlist):
    for ind, val in enumerate(df[replacecolumn]):
        if CheckValueExists(val, catlist) == False:
            df.replace(to_replace = val,inplace = True,value='Individual')
    df = RenameAllCategory(df,replacecolumn) 
    df.to_csv('banklist.csv')
    return df
    

# StdData function is to standardize the dataset []
# [] https://stackoverflow.com/questions/49641707/standardize-some-columns-in-python-pandas-dataframe
# [] https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
def StdData(df):
    # define standard scaler
    scaler = StandardScaler()
    # transform data
    df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'TRANSACTION_AMOUNT', 'BALANCE_AMT']] = scaler.fit_transform(df[['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'TRANSACTION_AMOUNT', 'BALANCE_AMT']])
    return df

# Encoder function transforms the character and object value in dataframe [5]
def Encoder(df): 
    # columnsToEncode is to get a list of columns having datatype as category and object
    #columnsToEncode = list(df.select_dtypes(include=['category','object']))
    #columnsToEncode = list(df['TRANSACTION_DETAILS'])
    # le is an instance of LabelEncoder with no parameter
    #le = LabelEncoder()
    # for loop will iterate over the columns. 
    # it will first try to use LabelEncoder to fit_transform
    # and if there are any error than the except will be executed
    #for feature in columnsToEncode:
    #    try:
    #        df.feature = le.fit_transform(df.feature)
    #    except:
    #        print('Error encoding ' + feature) 
    
    # cols variable is to store the column names where encoding is needed [1]     
    cols = ['TRANSACTION_DETAILS']
    # Encode labels of multiple columns at once
    df[cols] = df[cols].apply(LabelEncoder().fit_transform)
    return df

    
# References:
# [1] https://vitalflux.com/labelencoder-example-single-multiple-columns/
# [6] https://pythonbasics.org/seaborn-pairplot/
# [7] https://seaborn.pydata.org/generated/seaborn.heatmap.html

