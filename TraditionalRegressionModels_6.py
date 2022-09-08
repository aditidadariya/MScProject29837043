#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:44:03 2022

Student ID: 29837043
File: TraditionalRegressionModels_6.py

This file contains the traditional machine learning regression models to add bench mark for doing future time series forecasting.

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from datetime import datetime
import time
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Import AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
# Import SGDRegressor
from sklearn.linear_model import SGDRegressor
# Import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
# Import SVR
from sklearn.svm import SVR
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
# Import cross_validate function
from sklearn.model_selection import cross_validate
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV method
from sklearn.model_selection import GridSearchCV
# Importing sklearn
import sklearn
# Import train_test_split function
from sklearn.model_selection import train_test_split
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


####################### STEP 3: FEATURE SELECTION  ############################

# To implement any model on the data, we need to do feature selection first
# and therefore the dataset is devided into 2 parts.
# Firstly features (all the attributes except the target attribute) and secondly target (class attribute).
# The dfbankdataset dataframe has been split into dfbanktarget having only column "TRANSACTION_DETAILS", as this is the class attribute,
# and all the other attributes are taken into dfbankfeatures dataframe as features.

dfbankfeatures = dfbankdataset.drop("TRANSACTION_DETAILS", axis=1) # Features
dfbanktarget = dfbankdataset.loc[:, dfbankdataset.columns == "TRANSACTION_DETAILS"] # Target

# None of the feature selection methods are used here because the dataset is having very less number of attributes.
# and therefore, the dimentionality reduction has not been performed on this dataset.

# References:
# [] https://www.tutorialspoint.com/how-to-select-all-columns-except-one-in-a-pandas-dataframe#:~:text=To%20select%20all%20columns%20except%20one%20column%20in%20Pandas%20DataFrame,%5D.


"""# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ BUILDING BASIC MODELS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


############## STEP 4: SPLITING DATA WITH HOLDOUT VALIDATION ###################

# ================ 4.1. Split the data to train and test set ==================

# Data is splited into training and test set using the feature and target dataframes with holdout method
# to understand the model performance with 70% training set and 30% test set
X_train, X_test, Y_train, Y_test = train_test_split(dfbankfeatures, dfbanktarget, test_size=0.3, random_state=None)

print('Total records of Train dataset: ', len(X_train))
print('Total attributes of Train dataset: ', len(X_train.columns))
print('Total records of Test dataset: ', len(X_test))

# Give a new line to clearly format the output
Newline()

# Output:
# Output after DATE indexed
#Total records of Train dataset:  81340
#Total attributes of Train dataset:  5
#Total records of Test dataset:  34861

############## STEP 5: BUILD BASE REGRESION MODEL WITH HOLDOUT VALIDATION ###################

# Spot Check Algorithms with machine learning algorithms [8] [9] [10] [11] [12]
# Base Model is built below using the HoldOut Validation (train test split) to evaluate the root mean squared error

start_time = datetime.now()
#print("Basic Regression model start_time is - ", start_time)

# ========================== 5.1. Define models ===============================

# Clear lists
ClearLists()

# Give a new line to clearly format the output
Newline()

# Calling CreateRegModel function that creates model obejcts
CreateRegModel()

# ===================== 5.2. Build Model and Evaluate it  =====================

# Calling BasicRegModel function to build the basic models
basic_score = BasicRegModel(models, X_train, X_test, Y_train, Y_test)

# Create a dataframe to store accuracy
dfbasicscore = pd.DataFrame(basic_score)    
print(dfbasicscore)

# Give a new line to clearly format the output
Newline()

end_time = datetime.now()
#print("Basic Regression model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Basic Regression model for all accounts is {}".format(totaltime))

# Give a new line to clearly format the output
Newline()

# Output:
#  Model Name  Mean squared error  Root mean squared error   Coefficient of determination
#0         LR        6.641091e+02             2.577031e+01                   1.374279e-01
#1       SGDR        2.724666e+59             5.219833e+29                  -3.538908e+56
#2        SVR        8.119064e+02             2.849397e+01                  -5.453735e-02
#3        KNR        3.163420e+02             1.778601e+01                   5.891220e-01
#4        RFR        2.560411e+02             1.600128e+01                   6.674433e-01
#5        ABR        4.773765e+02             2.184895e+01                   3.799638e-01

# Total time to run the Basic Regression model for all accounts is 5:22:51.526451

# The output shows that the highest RMSE is for SGDR with 5.219833e+29 and lowest RMSE is for RFR with 1.600128e+01. 
# To improve the performance of each model, the evaluation is done with respect to different random_state in the next step
# Also cross validation is performed further to tune the models

################## STEP 6: BUILD MODELS WITH RANDOM STATE #####################

# Model is build with random state to evelaute the root mean squared error

# ======================= 6.1. Define models ==================================

# Clear lists
ClearLists()

# Calling CreateRegModel function that creates model obejcts
CreateRegModel()

# ===================== 6.2. Build Model and Evaluate it  =====================

# Calling BuildModelRS to evaluate the models with random states and return the root mean squared error
rsscore = BuildModelRS(models, rand_state, dfbankfeatures, dfbanktarget)

# Create a dataframe to store accuracy
dfrsscore = pd.DataFrame(rsscore)    
print(dfrsscore)

# Get the Random State having minimum RMSE
minrmse = dfrsscore["Root mean squared error"].min()
minmodel = dfrsscore.loc[dfrsscore["Root mean squared error"]==minrmse]
print("The {} model has minimum RMSE {} with a Random State of {}".format(', '.join(minmodel['Model Name'].values), float(minmodel['Root mean squared error'].values),int(minmodel['Random State'].values)))
minrs = int(minmodel['Random State'].values)

# Give a new line to clearly format the output
Newline()

# The output shows that the root mean squared error changes in each randon state.

# The RFR model has minimum RMSE 15.794247197542933 with a Random State of 5

#   Model Name  Random State  Mean squared error  Root mean squared error   Coefficient of determination  
#0          LR             1        6.724959e+02             2.593253e+01                   1.373820e-01   
#1          LR             3        6.680937e+02             2.584751e+01                   1.369703e-01  
#2          LR             5        6.688118e+02             2.586140e+01                   1.340051e-01  
#3          LR             7        6.736971e+02             2.595568e+01                   1.337567e-01  
#4        SGDR             1        1.429223e+60             1.195501e+30                  -1.833281e+57  
#5        SGDR             3        1.663339e+62             1.289705e+31                  -2.148668e+59  
#6        SGDR             5        1.823798e+62             1.350481e+31                  -2.361501e+59  
#7        SGDR             7        1.727939e+62             1.314511e+31                  -2.221793e+59  
#8         SVR             1        8.193923e+02             2.862503e+01                  -5.104359e-02  
#9         SVR             3        8.185829e+02             2.861089e+01                  -5.742849e-02  
#10        SVR             5        8.128970e+02             2.851135e+01                  -5.256015e-02  
#11        SVR             7        8.192032e+02             2.862173e+01                  -5.333571e-02  
#12        KNR             1        3.193042e+02             1.786909e+01                   5.904249e-01  
#13        KNR             3        3.182507e+02             1.783958e+01                   5.888904e-01  
#14        KNR             5        3.084672e+02             1.756323e+01                   6.005887e-01  
#15        KNR             7        3.184556e+02             1.784532e+01                   5.905282e-01  
#16        RFR             1        2.565922e+02             1.601849e+01                   6.708664e-01  
#17        RFR             3        2.536897e+02             1.592764e+01                   6.722889e-01  
#18        RFR             5        2.494582e+02             1.579425e+01                   6.769950e-01  
#19        RFR             7        2.606884e+02             1.614585e+01                   6.648056e-01  
#20        ABR             1        5.065003e+02             2.250556e+01                   3.503064e-01  
#21        ABR             3        4.702647e+02             2.168559e+01                   3.925217e-01  
#22        ABR             5        4.671931e+02             2.161465e+01                   3.950662e-01 
#23        ABR             7        4.829240e+02             2.197553e+01                   3.790538e-01    

################## STEP 7: OPTIMIZE MODEL USING CROSS VALIDATION ##############

# StratifiedKFold Cross Validation is utilized to improve the performance as it splits the data approximately in the same percentage [17] [18]
# StratifiedKFold Cross Validation is used because there are multiple classes, and the split of train and test set could be properly done

start_time = datetime.now()
#print("Basic Regression model start_time is - ", start_time)

# ============================ 7.1. Define models =============================

# Clear lists
ClearLists()

# Calling CreateRegModel function that creates model obejcts
CreateRegModel()

# ===================== 7.2. Build Model and Evaluate it  =====================

# Calling BuildModelCV function to get the root mean squared error, cross validation results and names of models used
cvscore, results, name = BuildModelCV(models, dfbankfeatures, dfbanktarget, minrs)
# Create a dataframe to store scores
dfcvscore = pd.DataFrame(cvscore)    
print(dfcvscore)

# Give a new line to clearly format the output
Newline()

# Get the best Model having minimum RMSE
minrmse = dfcvscore["Root mean squared error"].min()
minmodel = dfcvscore.loc[dfcvscore["Root mean squared error"]==minrmse]
print("The {} model has minimum RMSE {}".format(', '.join(minmodel['Model Name'].values), float(minmodel['Root mean squared error'].values)))

end_time = datetime.now()
#print("Basic Regression model end_time is - ", end_time)

# Print the total time spend to run the basic model
totaltime = end_time - start_time
print("Total time to run the Cross Validation Baseline Regression model for all accounts is {}".format(totaltime))

# Give a new line to clearly format the output
Newline()

# Output:
#   Model Name  Mean squared error  Root mean squared error  Coefficient of determination
#0         LR       -6.684728e+02            -2.585472e+01                   1.368196e-01 
#1       SGDR       -7.966895e+61            -6.976835e+30                  -1.028508e+59 
#2        SVR       -8.159514e+02            -2.856486e+01                  -5.361611e-02 
#3        KNR       -3.117396e+02            -1.765545e+01                   5.974594e-01 
#4        RFR       -2.536694e+02            -1.592662e+01                   6.724433e-01 
#5        ABR       -4.783208e+02            -2.186978e+01                   3.823558e-01 

# The SGDR model has minimum RMSE -6.976834583036508e+30
# Total time to run the Cross Validation Baseline Regression model for all accounts is 2:32:48.389473



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
