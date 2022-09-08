#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibVisual_3.py

This file contains all the functions to be called for data visualization

"""

######################## Importing necessary libraries ########################
import pandas as pd
# Import LabelEncoder to change the data types of attributes
#from sklearn.preprocessing import LabelEncoder
# Import Config.py file to get all the variables here
from Config_1 import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess_2 import *
#import os
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

########################## Declarations of Visualization Functions ##############################
# my_autopct is defined to get rid of 0% displayed in the pie plot
# This function is called in PlotPiechartPerAccount function
def my_autopct(pct): # https://stackoverflow.com/questions/34035427/conditional-removal-of-labels-in-pie-chart
    return ('%.2f' % pct) if pct > 20 else ''

# PlotPiechartPerAccount function is defined to plot a pie chart for each account
def PlotPiechartPerAccount(df, plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    for act, dfactg in dfgroupedact:
        dfactg.groupby(['TRANSACTION_DETAILS']).sum().plot(kind='pie', y=plotcolumn, shadow=False, figsize=(8,10), legend=False, 
                                  autopct=my_autopct, pctdistance=0.9, radius=1.2)
        plt.tight_layout()
   
# To visualize the data again, heat map is demonstrated below [7]
def DrawHeatmap(df):
    # Clear the plot dimensions
    plt.clf()
    # Plot Heat graph to visualise the correlation
    plt.figure(figsize=(15,10))
    matrix1=df.corr()
    plt.title('Bank User Transaction Correlation', size = 15)
    sns.heatmap(matrix1, vmax=10.8, square=True, cmap='YlGnBu', annot=True)

    # Give a new line to clearly format the output
    Newline()
    
    # Wait for the graph to be displayed
    time.sleep(20)

# SubPlotsforAmt defined to show the spikes for WITHDRAWAL_AMT, DEPOSIT_AMT and BALANCE_AMT
def SubPlotsforAmt(df):
    values = df.values
    # specify columns to plot
    groups = [2, 3, 4,5]
    i = 1
    # plot each column
    plt.figure(figsize=(16,12))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()
    
# PlotDataByAcct function is defined to Plot Boxplot grouped by ACCOUNT_NO
def PlotDataByAcct(df,groupbycolumn,plotcolumn):
    df.boxplot(by = groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
    #[] https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    plt.show()
    # Give a new line to clearly format the output
    Newline()

# PlotDataPerAcctCat function is defined to Plot BoxPlot grouped by ACCOUNT_NO and TRANSACTION_DETAILS
def PlotDataPerAcctCat(df,groupbycolumn,plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    # Iterate through each group of data in actcount to plot a box plot
    for act, dfactg in dfgroupedact:
        dfactg.boxplot(by =groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
        #[] https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        plt.show()
    # Give a new line to clearly format the output
    Newline()
    
# PlotBusinessDataPerAccount function is defined to plot the monthly data for each account
def PlotBusinessDataPerAccount(df, plotcolumn, ruletype):
    actcount = df.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()
    # Iterate through each group of data in actcount to plot monthly data
    for ind, eachact in actcount.items():
        dfacctmaxrecords = df.loc[df['ACCOUNT_NO']==ind, [plotcolumn]]
        dfacctmaxrecords.resample(rule = ruletype)[plotcolumn].max().plot(kind = "bar", figsize=(15,6))
        plt.title("Monthly data of " + str(ind))
        plt.show()

