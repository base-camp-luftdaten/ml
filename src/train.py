# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:43:25 2019

@author: Josua
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import requests 
import time
from pandas.io.json import json_normalize

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM #http://colah.github.io/posts/2015-08-Understanding-LSTMs/
from keras.layers import Dropout
from keras.models import model_from_json



def getSensorList():
    response = requests.get("http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/sensors")
    content = response.text
    splitResponse = [x.strip() for x in content.split('}')]
    sensorList = []
    for i in splitResponse:
        sensor = i[0:7]
        sensor = sensor.replace('"','')
        sensor = sensor.replace('{','')
        sensor = sensor.replace(':','')
        sensor = sensor.replace(',','')
        sensorList.append(sensor)
    return sensorList
sensorList = getSensorList()

def getDataFromSensor(index, timestamp):
    data = pd.read_json("http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/bySensor?sensor="+ index +"&timestamp="+timestamp,"index")
    return data
  
def trainFromSensors(number):
    i = 0
    sensorList = getSensorList()
    week = 7*24*60*60
    now = time.time()
    timestamp = int(now-week)
    while i < number:
        index = sensorList[i]
        data = getDataFromSensor(index, str(timestamp))
        training_set = np.column_stack((data['p10'],data['p25']))
        sc = MinMaxScaler(feature_range = (0,1))
        scaled_set = sc.fit_transform(training_set)    
        X,y = inAndOutput(scaled_set)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = loadModel("regressor.json","model.h5")
        model =furtherTraining(X_train,y_train,model)
        saveModel(model)
trainFromSensors(1)        

###############################################
def inAndOutput(training_set):
        X = []
        y = []
        for i in range(60, training_set.shape[0]-5):
            X.append(training_set[i-60:i, :])
            y.append(training_set[i:i+5, 0])
        X, y = np.array(X), np.array(y)
        # Reshaping
        y = np.reshape(y, (y.shape[0],y.shape[1]))
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        return X, y
    
def prepData(filepath):
    dataset_train = pd.read_csv(filepath)
    training_set = np.column_stack((dataset_train['P1'],dataset_train['P2']))
    # Feature Scaling
    
    sc = MinMaxScaler(feature_range = (0,1))
    scaled_set = sc.fit_transform(training_set)
    
    # Creating a data structure with 60 timesteps and 1 output
    
    X,y = inAndOutput(scaled_set)
    return X,y

X_train,y_train = prepData('../Data/trainwithp1p2.csv')
X_test,y_test = prepData('../Data/testwithp1p2.csv')
#########################################################
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
def myRegressor(X_train,y_train):
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50)) 
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = y_train.shape[1]))
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    #scores = regressor.evaluate(X_test,y_test,verbose=0)
    #print(scores)
    return regressor
regressor = myRegressor(X_train,y_train)

def loadModel(jsonpath,h5path):
    json_file = open(jsonpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5path)
    return loaded_model

def furtherTraining(X_train,y_train,regressor):
    regressor.fit(X_train,y_train, epochs = 100, batch_size = 32)
    return regressor

def saveModel(regressor):
    regressor_json = regressor.to_json()
    with open ("regressor.json", "w") as json_file:
        json_file.write(regressor_json)
    regressor.save_weights("model.h5")
saveModel(regressor)
#regressor = furtherTraining(X_test,y_test,regressor)
'''
nur mit einem datenset testen best loss = 0.01
mit zwei 0.0035
accuracy beidemal 0.2
wenn man mit einem trainiert und dem anderen tesete loss von 0.08

'''