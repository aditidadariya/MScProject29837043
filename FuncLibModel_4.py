#-#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: FuncLibModel_4.py

This file contains all the functions to be called for all data modelling 

"""

######################## Importing necessary libraries ########################
import pandas as pd
# Import Config.py file to get all the variables here
from Config_1 import *
# Import FuncLibPreProcess.py file
from FuncLibPreProcess_2 import *
# Import FuncLibVisual.py file
from FuncLibVisual_3 import *
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Import SGDRegressor
from sklearn.linear_model import SGDRegressor
# Import SVR
from sklearn.svm import SVR
# Import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Import AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

# Import mean_squared_error and r2_score
from sklearn.metrics import mean_squared_error, r2_score
import math 
from statistics import mean
from numpy import array
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import statistics
from statistics import mode
# Import sqrt from math
from math import sqrt
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
# Import cross_validate function
from sklearn.model_selection import cross_validate
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV method
from sklearn.model_selection import GridSearchCV

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import tensorflow as tf
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout #https://www.codegrepper.com/code-examples/whatever/NameError%3A+name+%27Dropout%27+is+not+defined
from keras.layers import Flatten
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

########################## Declarations of Regresion Model Functions ##############################    

# Create model objects [] https://towardsdatascience.com/how-to-build-your-first-machine-learning-model-in-python-e70fd1907cdd
# https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
def CreateRegModel():
    # Linear Algorithms ===========
    # Create model object of LinearRegression
    models.append(('LR', LinearRegression()))
    # Create model object of SGDRegressor
    models.append(('SGDR', SGDRegressor(max_iter=1000, tol=1e-3)))
    
    # Non linear Algorithms =========== 
    # Create model object of SVR
    models.append(('SVR', SVR(C=1.0, epsilon=0.2)))
    # Create model object of KNeighborsRegressor
    models.append(('KNR', KNeighborsRegressor(n_neighbors=2)))
    
    # Ensemble Algorithms =============  
    # Create model object of RandomForestRegressor
    models.append(('RFR', RandomForestRegressor(n_estimators=100, random_state=0)))
    # Create model object of AdaBoostRegressor
    models.append(('ABR', AdaBoostRegressor(n_estimators=100, random_state=0)))
    return models

# Define BasicRegModel function to evaluates the models
def BasicRegModel(models, X_train, X_test, Y_train, Y_test):
    # Evaluate each model in turns
    for name, model in models:
        # Train the model
        modelfit = model.fit(X_train,Y_train)
        # Predict the response for test dataset
        Y_predict = modelfit.predict(X_test)
        # Store the accuracy in results
        results.append(metrics.mean_squared_error(Y_test, Y_predict))
        # Store the model name in names
        names.append(name)
        # Store the name and accuracy in basic_score list
        basic_score.append({"Model Name": name, "Mean squared error": metrics.mean_squared_error(Y_test, Y_predict), 
                            "Root mean squared error": np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)),
                            "Coefficient of determination": r2_score(Y_test, Y_predict)})
    return basic_score

# Define BuildModelRS function to evaluate the models based on the random states [8-12]
# rand_state variable has been defined in the config.py file with values 1,3,5,7
# returns the Model Name, Random State, Mean squared error, Root mean squared error and Coefficient of determination of each model in score list
def BuildModelRS(models,rand_state,features,target):
    # Evaluate each model in turn
    for name, model in models:
        # for loop will train and predict the decision tree model on different random states
        for n in rand_state:
            # The training set and test set has been splited using the feature and target dataframes with different random_state
            X_train, X_test, Y_train, Y_test = train_test_split(features,target, test_size=0.3, random_state=n)
            # Train Decision Tree Classifer
            modelfit = model.fit(X_train,Y_train)
            # Predict the response for test dataset
            Y_predict = modelfit.predict(X_test)
            # Store the accuracy in results
            results.append(np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))
            # Store the model name in names
            names.append(name)
            # Store the Model Name, Random State and other results into score list
            score.append({"Model Name": name, "Random State": int(n), "Mean squared error": metrics.mean_squared_error(Y_test, Y_predict), 
                                "Root mean squared error": np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)),
                                "Coefficient of determination": r2_score(Y_test, Y_predict)})
    return score

# Define function BuildModelCV to evaluate models on the data and utilize cross validate along with StratifiedKFold
def BuildModelCV(models,features,target,randomstate):
    # Evaluate each model in turn  
    for name, model in models:
        # Define StratifiedKFold [17] [18]
        skfold = StratifiedKFold(n_splits=10, random_state=randomstate, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(features,target)
        # Evaluate each model with cross validation # https://scikit-learn.org/stable/modules/model_evaluation.html
        cv_results = cross_validate(model, features, target, cv=skfold, scoring=['neg_mean_squared_error','neg_root_mean_squared_error','r2']) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        # Store the accuracy in results
        results.append(cv_results)
        # Store the model name in names
        names.append(name)
        # Store the Model Name, Random State and other results into score list
        score.append({"Model Name": name, "Mean squared error": cv_results['test_neg_mean_squared_error'].mean(), 
                            "Root mean squared error": cv_results['test_neg_root_mean_squared_error'].mean(),
                            "Coefficient of determination": cv_results['test_r2'].mean()})
    return score, results, names

########################## Declarations of ARIMA Model Functions ##############################  

# ADFStationarityTest function is defined to test the time series data for its stationarity using Augmented Dickey-Fuller test
# https://www.hackdeploy.com/augmented-dickey-fuller-test-in-python/#:~:text=The%20adfuller%20function%20returns%20a,a%20dictionary%20of%20Critical%20Values.
def ADFStationarityTest(df):
    #Dickey-Fuller test:
    adfTest = adfuller(df, autolag='AIC')
    # dfResults variable to store summary of test
    dfResults = pd.Series(adfTest[0:4], index = ['ADF Test Statistics', 'p-value', '#Lag Used', 'Number of Observations Used'])
    pValue = adfTest[1]
    # Store summary into dfResults
    for key,value in adfTest[4].items():
        dfResults['Critical Value (%s)' %key] = value
    print('Augmented Dickey-Fuller Test Results:')
    print(dfResults)
    # Evaluate the p value
    if pValue <= 0.001:
        print("The time series data has not unit roots and hence it is stationary")
    else:
        print("The time series data has unit roots and hence it is not stationary")

# RollingStats function is defined to plot rolling mean and rolling standard deriation of 30 days
# https://github.com/llSourcell/Time_Series_Prediction/blob/master/Time%20Series.ipynb
def RollingStats(df):
    # Calculate the Rolling Mean and Rolling STD of 30 days
    rolmean = df.rolling(window = 30).mean()  #https://stackoverflow.com/questions/50482884/module-pandas-has-no-attribute-rolling-mean
    rolstd = df.rolling(window = 30).std()  #https://stackoverflow.com/questions/50482884/module-pandas-has-no-attribute-rolling-mean
    rcParams['figure.figsize']=(8,4)
    
    #Plot rolling Statistics
    orig = plt.plot(df, color = "blue", label = "Original")
    mean = plt.plot(rolmean, color = "red", label = "Rolling Mean")
    std = plt.plot(rolstd, color = "black", label = "Rolling Std")
    plt.legend(loc = "best")
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block = False)

# PlotCorrData function is defined to plot the correction plots
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# https://github.com/llSourcell/Time_Series_Prediction/blob/master/Time%20Series.ipynb
def PlotCorrData(df):
    plt.rcParams.update({'figure.figsize':(8,4), 'figure.dpi':120})
    df.plot()
    plt.show()
    # Plot autocorrelation
    plot_acf(df)
    plt.axhline( y = 0, linestyle = "--", color = "gray")
    plt.axhline( y= -2/np.sqrt(len(df)), linestyle = "--", color = "gray")
    plt.axhline(y = 2 /np.sqrt(len(df)), linestyle = "--", color = "gray")
    plt.show()
    # Plot Partial autocorrelation
    plot_pacf(df, method= "ols")
    plt.axhline( y = 0, linestyle = "--", color = "gray")
    plt.axhline( y= -2/np.sqrt(len(df)), linestyle = "--", color = "gray")
    plt.axhline(y = 2 /np.sqrt(len(df)), linestyle = "--", color = "gray")
    plt.show()

# EvaluateARIMATestModel function is defined to Evaluate the Basic ARIMA model along observed p,d and q values
def EvaluateARIMATestModel(df_train, df_test, p, d, q, ind):
    # Create the model object with the specified order of p,d,q
    model = ARIMA(df_train.dropna(), order=(p,d,q))
    # Fit the model
    model_fit = model.fit()
    # Print the summary of the model
    print(model_fit.summary())
    # Get prediction start and end dates
    start_date = len(df_train)
    end_date = len(df_train)+len(df_test)-1
    
    # Get the test predictions
    testpred = model_fit.predict(start=start_date, end=end_date,typ='levels').rename(ind)
    # Create new dataframe
    df_testpredictions = testpred.to_frame(name = 'TESTPRED_TRANSACTION_AMOUNT')
    # Set the index
    df_testpredictions.index = df_test.index
    
    # Get the residuals
    testresiduals = model_fit.resid
    # Calculate the Root Mean Squared Error
    testrmse = np.sqrt(np.mean(testresiduals**2))
    
    return testrmse, df_testpredictions

# EvaluateARIMAFutureModel function is defined to Evaluate the Basic ARIMA model along observed p,d and q values
def EvaluateARIMAFutureModel(df_train, df_test, p, d, q, ind):
    # Create the model object with the specified order of p,d,q
    model = ARIMA(df_train.dropna(), order=(p,d,q))
    # Fit the model
    model_fit = model.fit()
    # Print the summary of the model
    print(model_fit.summary())
    # Get prediction start and end dates
    pred_start_date = df_test.index[0]
    pred_end_date = df_test.index[-1]
    
    # Get the future predictions for next 3 months
    futurepred = model_fit.predict(start=pred_end_date, end=pred_end_date+3,typ='levels').rename(ind)
    # Get the future prediction dates
    index_future_date = pd.period_range(start=pred_end_date,periods=4 ,freq='M')  # https://stackoverflow.com/questions/57580072/changing-period-to-datetime
    # Set the future date index to the predictions
    futurepred.index = index_future_date
    # Converted predicted data into dataframe
    df_futurepredictions = futurepred.to_frame(name = 'FUTUREPRED_TRANSACTION_AMOUNT') # https://pandas.pydata.org/docs/reference/api/pandas.Series.to_frame.html
    
    # Get the residuals
    futureresiduals = model_fit.resid
    # Calculate the Root Mean Squared Error
    futurermse = np.sqrt(np.mean(futureresiduals**2))
    
    return futurermse, df_futurepredictions

# EvaluateTuneARIMAmodels function is defined to evaluate combinations of p, d and q values for an ARIMA model
def EvaluateTuneARIMAModels(df_train, df_test, ind, p_values, d_values, q_values):
    # Clear the list
    rmsetunelist.clear()
    # Traversings through each value of p
    for p in p_values:
        # Traversings through each value of q       
        for q in q_values:
            # Setting the order
            order = [p,d_values,q]
            # Calling EvaluateARIMATestModel function specific p,d,q values
            testrmse, df_testpredictions = EvaluateARIMATestModel(df_train, df_test, p, d_values, q,ind)
            # Calling EvaluateARIMAFutureModel function specific p,d,q values
            futurermse, df_futurepredictions = EvaluateARIMAFutureModel(df_train, df_test, p, d_values, q,ind)
            # Store the Order and RMSE for each Account in list
            rmsetuneline = {'ACCOUNT_NO': str(ind), 'ORDER': order, 'TESTRMSE': testrmse, 'FUTURERMSE': futurermse}
            # Append the rmsetuneline to a rmsetunelist
            rmsetunelist.append(rmsetuneline)
    
    # Creating dataframe containing ACCOUNT_NO, ORDER, TESTRMSE and FUTURERMSE
    df_tunearimarmse = pd.DataFrame(rmsetunelist)
    # Retrieving the minimum root mean sqaured error of test and future data
    mintestrmse = df_tunearimarmse['TESTRMSE'].min()
    minfuturermse = df_tunearimarmse['FUTURERMSE'].min()
    # Comparing the minimum root mean sqaured error of test and future data to the best RMSE
    if mintestrmse < minfuturermse:
        df_bestarima = df_tunearimarmse.loc[df_tunearimarmse['TESTRMSE'] == mintestrmse,['ACCOUNT_NO','ORDER']]
        df_bestarima[["RMSE"]] = mintestrmse
    else:
        df_bestarima = df_tunearimarmse.loc[df_tunearimarmse['FUTURERMSE'] == minfuturermse,['ACCOUNT_NO','ORDER']]
        df_bestarima[["RMSE"]] = minfuturermse
    return df_bestarima

########################## Declarations of VARMAX Model Functions ##############################  

# EvaluateVARMAXTestModel function is defined to evaluate the prediction over test set
def EvaluateVARMAXTestModel(df_train, df_test, p, q):
    # VARMAX model is build with training data and enforce_stationarity to state that there is stationairyt in the dataset
    var_model = VARMAX(df_train, order=(p,q),enforce_stationarity= True)
    fitted_model = var_model.fit(disp=False)
    print(fitted_model.summary())

    # Prediction on testset
    testpredict = fitted_model.get_prediction(start=len(df_train),end=len(df_train)+len(df_test)-1, freq = "M")
    df_testpredictions=testpredict.predicted_mean
    # Converting the recorded predictions into a dataframe
    df_testpredictions.columns=['TRANSACTION_DETAILS_TESTPRED','TRANSACTION_AMOUNT_TESTPRED']
    df_testpredictions.index = df_test.index
    # Evaluating the root mean squared error
    td_testrmse=math.sqrt(mean_squared_error(df_testpredictions['TRANSACTION_DETAILS_TESTPRED'],df_test['TRANSACTION_DETAILS']))
    ta_testrmse=math.sqrt(mean_squared_error(df_testpredictions['TRANSACTION_AMOUNT_TESTPRED'],df_test['TRANSACTION_AMOUNT']))
    return td_testrmse, ta_testrmse, df_testpredictions


# EvaluateVARMAXFutureModel function is defined to evaluate the furture prediction
def EvaluateVARMAXFutureModel(df_train, df_test, p, q):
    # VARMAX model is build with training data and enforce_stationarity to state that there is stationairyt in the dataset
    var_model = VARMAX(df_train, order=(p,q),enforce_stationarity= True)
    fitted_model = var_model.fit(disp=False)
    print(fitted_model.summary())

    # Creating index with future dates
    index_future_date = pd.period_range(start=df_test.index[-1],periods=4 ,freq='M') 

    # Prediction on future
    futurepredict = fitted_model.get_prediction(start=df_test.index[-1],end=df_test.index[-1]+4-1, freq = "M")
    df_futurepredictions=futurepredict.predicted_mean
    # Converting the recorded predictions into a dataframe
    df_futurepredictions.columns=['TRANSACTION_DETAILS_FUTUREPRED','TRANSACTION_AMOUNT_FUTUREPRED']
    df_futurepredictions.index = index_future_date
    # Evaluating the root mean squared error
    td_futurermse = np.sqrt(np.mean(df_futurepredictions['TRANSACTION_DETAILS_FUTUREPRED']**2))
    ta_futurermse = np.sqrt(np.mean(df_futurepredictions['TRANSACTION_AMOUNT_FUTUREPRED']**2))
    return td_futurermse, ta_futurermse, df_futurepredictions

# EvaluateTuneVARMAXModels function is defined to evaluate combinations of p, d and q values for an VARMAX model
def EvaluateTuneVARMAXModels(df_train, df_test, ind, p_values, q_values):
    # Clear the list
    rmsetunelist.clear()
    # Traversings through each value of p
    for p in p_values:
        # Traversings through each value of q       
        for q in q_values:
            # Setting the order
            order = [p,q]
            #order = [p,d_values,q]
            # Calling EvaluateVARMAXTestModel function specific p,d,q values
            td_testrmse, ta_testrmse, df_testpredictions = EvaluateVARMAXTestModel(df_train, df_test, p, q)
            # Calling EvaluateVARMAXFutureModel function specific p,d,q values
            td_futurermse, ta_futurermse, df_futurepredictions = EvaluateVARMAXFutureModel(df_train, df_test, p, q)
            # Store the Order and RMSE for each Account in list
            rmsetuneline = {"CLUSTER": ind, 'ORDER': order, "TD_TESTRMSE": td_testrmse, "TD_FUTURERMSE": td_futurermse, "TA_TESTRMSE": ta_testrmse, "TA_FUTURERMSE": ta_futurermse}
            # Append the rmsetuneline to a rmsetunelist
            rmsetunelist.append(rmsetuneline)
    
    # Creating dataframe containing ACCOUNT_NO, ORDER, TESTRMSE and FUTURERMSE
    df_tunevarmaxrmse = pd.DataFrame(rmsetunelist)
    # Retreive the minimum root mean squared error
    rmseind = df_tunevarmaxrmse[['TD_TESTRMSE','TD_FUTURERMSE','TA_TESTRMSE','TA_FUTURERMSE']].idxmin()
    df_bestvarmax = df_tunevarmaxrmse.loc[df_tunevarmaxrmse.index == rmseind[0],['CLUSTER','ORDER']]
    return df_bestvarmax


########################## Declarations of Common Model Functions for LSTM and CNN ##############################  

# TrainTestData function is defined to split the dataset into train and test arrays
def SplitTrainTestData(df):
    #Spliting the data to 70/30 ratio for train and test data according to the indexes
    df_train = df[0:(int(len(df)*0.7))]
    df_test = df[(int(len(df)*0.7))-1:-1]
    # Convert the train data and test data into array
    trainingset = df_train.to_numpy()
    testset = df_test.to_numpy()
    return trainingset, testset, df_train, df_test

# ProduceGenerator function is defined to get the train and test generators
def ProduceGenerator(trainingset, testset, n_input):
    # Creating train and test generators
    traingenerator = TimeseriesGenerator(trainingset, trainingset, length=n_input, sampling_rate=1, batch_size=1)
    testgenerator = TimeseriesGenerator(testset, testset, length=n_input, sampling_rate=1, batch_size=1)
    return traingenerator, testgenerator

# CompileFitModel function is defined to compile and fit the LSTM Model
def CompileFitModel(model, traingenerator, testgenerator):
    # Defined the EarlyStopping if the loss is not impacted
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5) # https://keras.io/api/callbacks/early_stopping/
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # Display the summary of model compiled
    model.summary()
    # Evaluate the model
    model.fit_generator(generator=traingenerator, callbacks=[es], epochs=10, steps_per_epoch = 500, validation_data=testgenerator)
    return model

# Define CompileFitTuneModel function to evaluate the model and get the accuracy
def CompileFitTuneModel(model, traingenerator, testgenerator, ind, trainingset):
    # Clear list
    scores_list.clear()
    # Defined the EarlyStopping if the loss is not impacted
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5) # https://keras.io/api/callbacks/early_stopping/
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # Display the summary of model compiled
    model.summary()
    # Create a dataframe to store EPOCH and RMSE for each account
    dfepochslstm = pd.DataFrame()
    # Defined steps per epoch
    stepsperepoch = 500
    for n in eps:
        if len(trainingset) > stepsperepoch:
            model.fit_generator(generator=traingenerator, callbacks=[es], epochs=n, steps_per_epoch = stepsperepoch, validation_data=testgenerator, verbose = 0)
        else:
            model.fit_generator(generator=traingenerator, callbacks=[es], epochs=n, validation_data=testgenerator, verbose = 0)
        # Evaluate the model
        scores = model.history.history['root_mean_squared_error']
        # Store the Metrics Name and Accuracy into scores_list list
        scores_list.append({"ACCOUNT_NO": ind, "Epochs": n, "RMSE": scores[1]})
    
    # Define a dataframe to with scores_list
    dfepochslstm = pd.DataFrame(scores_list) 
    return dfepochslstm

# TuneRandomSearch function is defined to tune the model using RandomSearch and get best hyperparameters 
def TuneRandomSearch(trainingset, testset, modelhp, best_epoch):
    # Calling RandomSearch method to train LSTM model
    tuner = RandomSearch(modelhp, objective = 'loss', max_trials=5, executions_per_trial=3)
    # Defined the EarlyStopping if the loss is not impacted
    es = EarlyStopping(monitor='loss', mode='min', patience=5, verbose=1) # https://keras.io/api/callbacks/early_stopping/
    # Train the model
    tuner.search(trainingset,trainingset,epochs=best_epoch,validation_data=(testset,testset), callbacks=[es])
    # Retrieve the best_model
    best_model = tuner.get_best_models(num_models=1)[0]
    # Get the loss and mse from the best model
    loss, rmse = best_model.evaluate(testset, testset)
    # Give a new line to clearly format the output
    Newline()
    return tuner

# FinalCompileFitModel function is defined to compile and fit the LSTM Model with best_epoch and best_learning_rate
def FinalCompileFitModel(model, traingenerator, testgenerator, best_epoch, best_learning_rate):
    # Defined the EarlyStopping if the loss is not impacted
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5) # https://keras.io/api/callbacks/early_stopping/
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=best_learning_rate),metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # Display the summary of model compiled
    model.summary()
    # Evaluate the model
    model.fit_generator(generator=traingenerator, callbacks=[es], epochs=best_epoch, steps_per_epoch = 500, validation_data=testgenerator)
    return model


# PlotLosses function is defined to plot the loss in LSTM model
def PlotLosses(model):
    loss_per_epoch = model.history.history['loss']
    plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    plt.show()

# TestPrediction function is defined to create future predictions using testset
def TestPrediction(trainingset, testset, df_test, model):
    # Creating the current batch array to be used in predicting the future
    firstbatch = trainingset[-n_input:]
    currentbatch = firstbatch.reshape((1, n_input, n_features))
    testpredictions.clear()
    # Predict the furture using testset
    for i in range(len(testset)):
        # get the prediction value for the first batch
        currentpred = model.predict(currentbatch,verbose=0)[0]
        # append the prediction into the array
        testpredictions.append(currentpred) 
        # use the prediction to update the batch and remove the first value
        currentbatch = np.append(currentbatch[:,1:,:],[[currentpred]],axis=1)
    # Transform the testpredictions into the original form and append it in the dataframe
    testpred = pd.Series(testpredictions).array.astype(float)
    # Convert the testpred into dataframe
    df_testpredictions = pd.DataFrame(testpred, columns=['TESTPRED_TRANSACTION_AMOUNT'])
    df_testpredictions.index = df_test.index
    # Evaluate the root mean squared error
    rmse = np.sqrt(np.mean(df_testpredictions['TESTPRED_TRANSACTION_AMOUNT']**2))
    return rmse, df_testpredictions

# FuturePrediction function is defined to create future predictions using testset
def FuturePrediction(trainingset, testset, df_test, model):
    # Creating the current batch array to be used in predicting the future
    firstbatch = testset[-n_input:]
    currentbatch = firstbatch.reshape((1, n_input, n_features))
    # Clear the testpredictions list
    futurepredictions.clear()
    # Retreiving the Start and End Index Dates from test data
    pred_start_date = df_test.index[0]
    pred_end_date = df_test.index[-1]
    # Creating index with future dates
    index_future_date = pd.period_range(start=pred_end_date,periods=4 ,freq='M') 
    # Predict the furture using future index
    for i in range(len(index_future_date)):
        # get the prediction value for the first batch
        currentpred = model.predict(currentbatch,verbose=0)[0]
        # append the prediction into the array
        futurepredictions.append(currentpred) 
        # use the prediction to update the batch and remove the first value
        currentbatch = np.append(currentbatch[:,1:,:],[[currentpred]],axis=1)
    # Convert the futurepred into series
    futurepred = pd.Series(futurepredictions).array.astype(float)
    # Converted predicted data into dataframe
    df_futurepredictions = pd.DataFrame(futurepred, columns=['FUTUREPRED_TRANSACTION_AMOUNT'])
    df_futurepredictions.index = index_future_date
    # Evaluet the Root Mean Squared Error
    rmse = np.sqrt(np.mean(df_futurepredictions['FUTUREPRED_TRANSACTION_AMOUNT']**2))
    return rmse, df_futurepredictions

#https://stackoverflow.com/questions/56352709/making-matplotlib-pyplot-plot-work-with-periodindex
# PlotTestFuturePrediction function is defined to plot the predictions
def PlotTestFuturePrediction(df1,df2,df3,ind):
    fig = plt.figure(figsize=(8,4))
    fig, ax = plt.subplots()
    df1.plot(ax=ax)
    df2.plot(ax=ax)
    df3.plot(ax=ax)
    plt.legend(loc = 'best')
    plt.title("Predicted Data of " + str(ind))

# MultiCompileFitModel function is defined to fit the LSTM Multivariate models
def MultiCompileFitModel(model, generator, best_epoch, best_learning_rate):
    # Defined the EarlyStopping if the loss is not impacted
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5) # https://keras.io/api/callbacks/early_stopping/
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=best_learning_rate),metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # Display the summary of model compiled
    model.summary()
    # Evaluate the model
    model.fit_generator(generator=generator, callbacks=[es], epochs=best_epoch, steps_per_epoch = 500)
    return model

# MultiTestPrediction function is defined to make predictions over multivariate test set
def MultiTestPrediction(train_dataset, df_test, n_input, n_features, model):
    testpredictions.clear()
    # Make a multi step prediction out of sample
    for i in range(len(df_test)):
        if i != len(df_test)-1:
            x_input = array([df_test.iloc[i].values,df_test.iloc[i+1].values]).reshape((1, n_input, n_features))
            yhat = model.predict(x_input, verbose=0)[0]
            testpredictions.append(yhat)
    df_testpredictions = pd.DataFrame(testpredictions, columns=[['TESTPRED_TRANSACTION_DETAILS','TESTPRED_TRANSACTION_AMOUNT']])
    df_testpredictions.index = df_test.index[:-1]
    # Evaluate the root mean squared error
    td_testrmse=math.sqrt(mean_squared_error(df_testpredictions['TESTPRED_TRANSACTION_DETAILS'],df_test['TRANSACTION_DETAILS'].iloc[1:]))
    ta_testrmse=math.sqrt(mean_squared_error(df_testpredictions['TESTPRED_TRANSACTION_AMOUNT'],df_test['TRANSACTION_AMOUNT'].iloc[1:]))
    return td_testrmse, ta_testrmse, df_testpredictions

# MultiFuturePrediction function is defined to make predictions over multivariate future of 4 months
def MultiFuturePrediction(train_dataset, df_test, n_input, n_features, model):
    futurepredictions.clear()
    # Creating index with future dates
    index_future_date = pd.period_range(start=df_test.index[-1],periods=4 ,freq='M') 
    x_input = array([df_test.iloc[-2].values,df_test.iloc[-1].values]).reshape((1, n_input, n_features))
    # Predict the furture using future index
    for i in range(len(index_future_date)):
        yhat = model.predict(x_input, verbose=0)[0]
        x_input = array([df_test.iloc[i].values,df_test.iloc[i+1].values]).reshape((1, n_input, n_features))
        futurepredictions.append(yhat)
    df_futurepredictions = pd.DataFrame(futurepredictions, columns=[['FUTUREPRED_TRANSACTION_DETAILS','FUTUREPRED_TRANSACTION_AMOUNT']])
    df_futurepredictions.index = index_future_date
    # Evaluate the root mean squared error
    td_futurermse=np.sqrt(np.mean(df_futurepredictions['FUTUREPRED_TRANSACTION_DETAILS']**2))
    ta_futurermse=np.sqrt(np.mean(df_futurepredictions['FUTUREPRED_TRANSACTION_AMOUNT']**2))
    return td_futurermse, ta_futurermse, df_futurepredictions


########################## Declarations of LSTM Model Functions ##############################

# CreateLSTMModel function is defined to create a LSTM model object
def CreateLSTMModel(trainingset, n_input, n_features):
    # Create a model object for LSTM
    model = Sequential()
    # Define the LSTM layers
    model.add(LSTM(64, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    # Define the output layer
    model.add(Dense(trainingset.shape[1]))
    return model

# CreateLSTMModelHP function is defined to create and compile the LSTM model with the hyperparameters [26]
def CreateLSTMModelHP(hp):
    # Define Sequential model
    model = Sequential()
    # To get the best value of number of neurons per hidden layer, hp.Int hyperparameter is used 
    # by giving a range from min_value as 32 and max_value as 300 with the step size as 50
    model.add(LSTM(units= hp.Int('units', min_value=32, max_value=300, step=50), activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units= hp.Int('units', min_value=32, max_value=300, step=50), activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    # Define the output layer
    model.add(Dense(1))
    # compile model to get the best Adam optimizer. learning_rate from 0.01 and 0.001 is used
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-4])),metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# FinalLSTMModel function is defined to create the final lstm model with best_units received
def FinalLSTMModel(trainingset, n_input, n_features, best_units):
    # Constructing units to be used in second LSTM layers
    secunits = int(best_units/2)
    # Define Sequential model
    model = Sequential()
    # Define the LSTM layers
    model.add(LSTM(units=best_units, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=secunits, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    # Define the output layer
    model.add(Dense(trainingset.shape[1]))
    return model

########################## Declarations of CNN Model Functions ##############################

# ReshapeCNNData function is defined to reshape the input train and test datasets
def ReshapeCNNData(trainingset, testset):
    trainConv2D = trainingset.reshape((trainingset.shape[0], 1, n_features))
    testConv2D = testset.reshape((testset.shape[0], 1, n_features))
    return trainConv2D, testConv2D

# CreateCNNModel function is defined to create CNN layers
def CreateCNNModel(training, n_input, n_features):
    # Defining the Conv2D layers
    model = Sequential()
    # Define the CNN layers
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu', batch_size = 1, 
                     input_shape=(n_input,1, n_features) )) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    # Define the output layer
    model.add(Dense(units=1))
    return model

# CreateCNNModelHP function is defined to create and compile the CNN model with the hyperparameters [26]
def CreateCNNModelHP(hp):
    # Defining the Conv2D layers
    model = Sequential()
    # To get the best value of number of neurons per hidden layer, hp.Int hyperparameter is used 
    # by giving a range from min_value as 32 and max_value as 300 with the step size as 50
    model.add(Conv2D(filters= hp.Int('units', min_value=32, max_value=300, step=50), kernel_size=(3,3), padding="same", activation='relu', batch_size = 1, 
                     input_shape=(n_input,1, n_features) )) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(filters= hp.Int('units', min_value=32, max_value=300, step=50), kernel_size=(3,3), padding="same", activation='relu')) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(units=1))
    # compile model - To get the best Adam optimizer, learning_rate from 0.01 and 0.001 is used
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-4])),metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# FinalCNNModel function is defined to create the final cnn model with best_units received
def FinalCNNModel(trainingset, n_input, n_features, best_units):
    # Constructing units to be used in second CNN layers
    secunits = int(best_units/2)
    # Define Sequential model
    model = Sequential()
    # Define CNN layers
    model.add(Conv2D(filters=best_units, kernel_size=(3,3), padding="same", activation='relu', batch_size = 1, 
                     input_shape=(n_input,1, n_features) )) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Conv2D(filters=secunits, kernel_size=(3,3), padding="same", activation='relu')) #, return_sequences=True))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(units=1))
    return model

