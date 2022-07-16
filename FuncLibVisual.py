#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibVisual.py

This file contains all the functions to be called by mainsfile.py

"""

######################## Importing necessary libraries ########################
import pandas as pd
# Import LabelEncoder to change the data types of attributes
#from sklearn.preprocessing import LabelEncoder
# Import Config.py file to get all the variables here
from Config import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess import *
#import os
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
#from sklearn.preprocessing import StandardScaler

########################## Declarations of Visualization Functions ##############################

"""
# PlotPairplot function is defined to draw a pairplot to visualize the data [6]
def DrawPairplot(df):
    # Clear the plot dimensions
    plt.clf()
    
    # Plot the multivariate plot using pairplot
    sns.set(style="ticks", color_codes=True)
    #grh = sns.pairplot(dfbankdata,diag_kind="kde")
    sns.pairplot(df,diag_kind="kde")
    plt.show()
    
    # Save the pairplot in png file
    plt.savefig("pairplot.png")
    
    # Give a new line to clearly format the output
    Newline()
    
    # Wait until the graph is displayed
    time.sleep(20)
 """
   
# To visualize the data again, heat map is demonstrated below [7]
def DrawHeatmap(df):
    # Clear the plot dimensions
    plt.clf()
    
    # Plot Heat graph to visualise the correlation
    plt.figure(figsize=(15,10))
    matrix1=df.corr()
    plt.title('Bank User Transaction Correlation', size = 15)
    sns.heatmap(matrix1, vmax=10.8, square=True, cmap='YlGnBu', annot=True)
    
    # Save the heatplot in png file
    plt.savefig("heatplot.png")
    
    # Give a new line to clearly format the output
    Newline()
    
    # Wait for the graph to be displayed
    time.sleep(20)

"""
# DrawBoxplot function is defined to visualize the pattern for each account
def DrawBoxplot(df):
    # Clear the plot dimensions
    plt.clf()
    
    # Plot Boxplot to visualise the correlation between ACCOUNT_NO, WITHDRAWAL_AMT and DEPOSIT_AMT attributes [] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html
    sns.set_theme(style="whitegrid")
    #df.boxplot(by ='ACCOUNT_NO', column =['WITHDRAWAL_AMT','DEPOSIT_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=12)
    df.boxplot(by ='ACCOUNT_NO', column =['WITHDRAWAL_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=12)
    plt.suptitle('')
    # Save the heatplot in png file
    plt.savefig("boxplotwithdrawal.png")
    df.boxplot(by ='ACCOUNT_NO', column =['DEPOSIT_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=12)
    plt.suptitle('')
    # Save the heatplot in png file
    plt.savefig("boxplotdeposit.png")
    # Plot Boxplot to visualise the correlation between ACCOUNT_NO and BALANCE_AMT
    df.boxplot(by ='ACCOUNT_NO', column =['BALANCE_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=12)
    plt.suptitle('')
    # Save the heatplot in png file
    plt.savefig("boxplotbalance.png")
    
    # Give a new line to clearly format the output
    Newline()
    
    # Wait for the graph to be displayed
    time.sleep(20)
"""
"""
# BoxPlotPerAcct function is defined to draw BoxPlot for each ACCOUNT_NO [] https://stackoverflow.com/questions/68734504/boxplot-by-two-groups-in-pandas
def BoxPlotPerAcct(df):
    # Clear the plot dimensions
    plt.clf()
    dfgroupedact = df.groupby('ACCOUNT_NO')
    for act, dfactg in dfgroupedact:
        #print(act)
        #print (dfactg)
        # Clear the plot dimensions
        plt.clf()
        # Plot Boxplot to visualise the correlation between TRANSACTION_DETAILS, WITHDRAWAL_AMT, DEPOSIT_AMT and BALANCE_AMT attributes [] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html
        sns.set_theme(style="whitegrid")
        dfactg.boxplot(by ='TRANSACTION_DETAILS', column =['WITHDRAWAL_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=16)
        plt.suptitle('')
        # Save the heatplot in png file
        plt.savefig("boxplotwithdrawal_" + act + ".png")
        plt.show()
        dfactg.boxplot(by ='TRANSACTION_DETAILS', column =['DEPOSIT_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=16)
        plt.suptitle('')
        # Save the heatplot in png file
        plt.savefig("boxplotdeposit_" + act + ".png")
        plt.show()
        dfactg.boxplot(by ='TRANSACTION_DETAILS', column =['BALANCE_AMT'],layout=(4,1),figsize=(10,30),grid = True,rot=45,fontsize=16)
        plt.suptitle('')
        # Save the heatplot in png file
        plt.savefig("boxplotbalance_" + act + ".png")
        plt.show()
"""

# PlotDataByAcct function is defined to Plot Boxplot grouped by ACCOUNT_NO
def PlotDataByAcct(df,groupbycolumn,plotcolumn):
    #dfgroupedact = df.groupby('ACCOUNT_NO')
    #for act, dfactg in df:
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(kind = 'line',legend=False,layout=(4,2),figsize=(12,4))
    df.boxplot(by = groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=False,layout=(4,1),figsize=(12,4))
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=False,xlim=['2017-01-01','2019-01-01'],ylim=[0,20],layout=(4,1),figsize=(12,4))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #[] https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    plt.show()
    plt.savefig("Boxplot_" + groupbycolumn + "_" + plotcolumn + ".png")
    
    # Give a new line to clearly format the output
    Newline()
    
    # Wait for the graph to be displayed
    time.sleep(20)
    
def PlotDataPerAcctCat(df,groupbycolumn,plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    for act, dfactg in dfgroupedact:
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(kind = 'line',legend=False,layout=(4,2),figsize=(12,4))
        dfactg.boxplot(by =groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=False,layout=(4,1),figsize=(12,4))
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=False,xlim=['2017-01-01','2019-01-01'],ylim=[0,20],layout=(4,1),figsize=(12,4))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #[] https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        plt.show()
        plt.savefig("plot_" + act + "_" + plotcolumn + ".png")
    
    # Give a new line to clearly format the output
    Newline()
    
    # Wait for the graph to be displayed
    time.sleep(20)
    
"""
# PlotData function is defined to plot the data 
def PlotData(df,groupbycolumn,plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    for act, dfactg in dfgroupedact:
        dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=True,layout=(4,1),figsize=(5,5))
        #dfactg.groupby(groupbycolumn)[plotcolumn].plot(legend=False,xlim=['2017-01-01','2019-01-01'],ylim=[0,20],layout=(4,1),figsize=(12,4))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #[] https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        plt.show()
        plt.savefig("plot_" + act + "_" + plotcolumn + ".png")
"""
