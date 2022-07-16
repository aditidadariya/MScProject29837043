#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibModel.py

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
# Import FuncLibVisual.py file
from FuncLibVisual import *
#import os
import requests
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import time

# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import Ridge
from sklearn.linear_model import Ridge
# Import Lasso
from sklearn.linear_model import Lasso
# Import SGDRegressor
from sklearn.linear_model import SGDRegressor
# Import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
# Import LinearSVR
from sklearn.svm import LinearSVR
# Import SVR
from sklearn.svm import SVR
# Import mean_squared_error and r2_score
from sklearn.metrics import mean_squared_error, r2_score
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import StandardScaler to Standardize the data
#from sklearn.preprocessing import StandardScaler


########################## Declarations of Model Functions ##############################    

# Create model objects [] https://towardsdatascience.com/how-to-build-your-first-machine-learning-model-in-python-e70fd1907cdd
def CreateModelObject():
    # Create model object of DecisionTreeRegressor
    models.append(('DTR', DecisionTreeRegressor(random_state=0)))
    # Create model object of LinearRegression
    models.append(('LR', LinearRegression()))
    # Create model object of RandomForestRegressor
    #models.append(('RFR', RandomForestRegressor(max_depth=2, random_state=42)))
    models.append(('RFR', RandomForestRegressor(random_state=0)))
    # Create model object of Ridge
    #models.append(('RR', Ridge(alpha=1.0)))            #Not needed
    # Create model object of Lasso
    #models.append(('LsR', Lasso(alpha=1.0)))            #Not needed
    # Create model object of SGDRegressor
    #models.append(('SGDR', SGDRegressor(max_iter=1000, tol=1e-3)))
    models.append(('SGDR', SGDRegressor()))
    # Create model object of KNeighborsRegressor
    #models.append(('KNR', KNeighborsRegressor(n_neighbors=2)))
    models.append(('KNR', KNeighborsRegressor()))
    # Create model object of LinearSVR
    #models.append(('LSVR', LinearSVR(random_state=0, tol=1e-5)))
    models.append(('LSVR', LinearSVR(random_state=0)))
    # Create model object of SVR
    #models.append(('SVR', SVR(C=1.0, epsilon=0.2)))
    models.append(('SVR', SVR()))
    return models

# Define BasicModel function to evaluates the models
# displays the confusion matrix and plot it [13] [14]
# displays the classification_report
def BasicModel(models, X_train, X_test, Y_train, Y_test):
    # Evaluate each model in turns
    for name, model in models:
        # Train the model
        modelfit = model.fit(X_train,Y_train)
        # Predict the response for test dataset
        Y_predict = modelfit.predict(X_test)
        # Store the accuracy in results
        #results.append(metrics.accuracy_score(Y_test, Y_predict))
        results.append(metrics.mean_squared_error(Y_test, Y_predict))
        # Store the model name in names
        names.append(name)
        # Print the prediction of test set
        # print('On %s Accuracy is: %f ' % (name, metrics.accuracy_score(Y_test, Y_predict)*100))
        #if name == 'LR' or name == 'RR' or name == "LsR" or name == "SGDR" or name == "LSVR":
        #    print('On %s Coeffiecients are: ' % name)
        #    print(modelfit.coef_)
        #print('On %s Mean squared error is: %.2f ' % (name, metrics.mean_squared_error(Y_test, Y_predict)))
        #print('On %s Root mean squared error is: %.2f ' % (name, np.sqrt(mean_squared_error(Y_test, Y_predict))))
        #print('On %s Coefficient of determination is: %.2f ' % (name, r2_score(Y_test, Y_predict)))
        
        # Store the name and accuracy in basic_score list
        #basic_score.append({"Model Name": name, "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
        basic_score.append({"Model Name": name, "Mean squared error": metrics.mean_squared_error(Y_test, Y_predict), 
                            "Root mean squared error": np.sqrt(mean_squared_error(Y_test, Y_predict)),
                            "Coefficient of determination": r2_score(Y_test, Y_predict)})
        # Plot the Actual and Predicted values
       # plt.plot(Y_test, Y_predict, '.')
       # x = np.linspace(0, 60, 10)
       # y = x
       # plt.plot(x, y)
       # plt.show()
        
        # Print Confusion Matrix and Classification Report
        #print(confusion_matrix(Y_test, Y_predict))
        #print(classification_report(Y_test, Y_predict))
        # Plot Confusion Matrix [13] [14]
        #cm = confusion_matrix(Y_test, Y_predict, labels=modelfit.classes_)
        #cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelfit.classes_)
        #cmdisp = ConfusionMatrixDisplay(confusion_matrix=Coefficients, display_labels=modelfit.coef_)
        #cmdisp.plot()
        #plt.show()
    return basic_score

