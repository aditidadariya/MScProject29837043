#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibVisual_3.py

This file contains all the functions to be called for data visualization

"""

######################## Importing necessary libraries ########################
import pandas as pd
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
# my_autopct is defined to get rid of 0% displayed in the pie plot [1]
# This function is called in PlotPiechartPerAccount function
def my_autopct(pct):
    return ('%.2f' % pct) if pct > 20 else ''

# PlotPiechartPerAccount function is defined to plot a pie chart for each account [2] [3]
def PlotPiechartPerAccount(df, plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    for act, dfactg in dfgroupedact:
        dfactg.groupby(['TRANSACTION_DETAILS']).sum().plot(kind='pie', y=plotcolumn, shadow=False, figsize=(8,10), legend=False, 
                                  autopct=my_autopct, pctdistance=0.9, radius=1.2)
        plt.tight_layout()
   
# To visualize the data again, heat map is demonstrated below [4] 
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

# SubPlotsforAmt defined to show the spikes for WITHDRAWAL_AMT, DEPOSIT_AMT TRANSACTION_AMOUNT and BALANCE_AMT [5] [6]
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
    
# PlotDataByAcct function is defined to Plot Boxplot grouped by ACCOUNT_NO [7] [8] [9]
def PlotDataByAcct(df,groupbycolumn,plotcolumn):
    df.boxplot(by = groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
    plt.show()
    # Give a new line to clearly format the output
    Newline()

# PlotDataPerAcctCat function is defined to Plot BoxPlot grouped by ACCOUNT_NO and TRANSACTION_DETAILS [7] [8] [10]
def PlotDataPerAcctCat(df,groupbycolumn,plotcolumn):
    dfgroupedact = df.groupby('ACCOUNT_NO')
    # Iterate through each group of data in actcount to plot a box plot
    for act, dfactg in dfgroupedact:
        dfactg.boxplot(by =groupbycolumn, column =[plotcolumn],layout=(4,1),figsize=(16,12),grid = True,rot=45,fontsize=12)
        plt.show()
    # Give a new line to clearly format the output
    Newline()
    
# PlotBusinessDataPerAccount function is defined to plot the monthly data for each account [11] [12]
def PlotBusinessDataPerAccount(df, plotcolumn, ruletype):
    actcount = df.groupby(['ACCOUNT_NO'])['ACCOUNT_NO'].count()
    # Iterate through each group of data in actcount to plot monthly data
    for ind, eachact in actcount.items():
        dfacctmaxrecords = df.loc[df['ACCOUNT_NO']==ind, [plotcolumn]]
        dfacctmaxrecords.resample(rule = ruletype)[plotcolumn].max().plot(kind = "bar", figsize=(15,6))
        plt.title("Monthly data of " + str(ind))
        plt.show()


################################# References ##################################

# [1] https://stackoverflow.com/questions/34035427/conditional-removal-of-labels-in-pie-chart
# [2] https://www.delftstack.com/howto/python-pandas/python-to-plot-pie-chart-and-table-of-pandas-dataframe/
# [3] https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
# [4] https://seaborn.pydata.org/generated/seaborn.heatmap.html
# [5] https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# [6] https://stackoverflow.com/questions/31726643/how-to-plot-in-multiple-subplots
# [7] https://stackoverflow.com/questions/68734504/boxplot-by-two-groups-in-pandas
# [8] https://www.geeksforgeeks.org/box-plot-visualization-with-pandas-and-seaborn/
# [9] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html
# [10] https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.boxplot.html
# [11] https://github.com/krishnaik06/Live-Time-Series/blob/main/Time%20Series%20EDA.ipynb
# [12] https://stackoverflow.com/questions/17001389/pandas-resample-documentation

