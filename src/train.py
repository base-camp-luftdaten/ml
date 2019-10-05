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
import urllib.request, json
import time
import os
from pandas.io.json import json_normalize

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM #http://colah.github.io/posts/2015-08-Understanding-LSTMs/
from keras.layers import Dropout
from keras.models import model_from_json


import matplotlib.pyplot as plt

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
#sensorList = getSensorList()

'''
Starting from timestamp, retrieves all measurements
made by the sensor in 1 hour steps, up to now
'''
def getDataFromSensor(sensorID, timestamp):
    data = []
    fullUrl = "http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/bySensorUntilNow?sensor=" + sensorID + "&timestamp=" + str(timestamp)
    print(fullUrl)
    with urllib.request.urlopen(fullUrl) as url:
        fullData = json.loads(url.read().decode())
        for singleResult in fullData:
            measurement = singleResult['measurement']
            weatherReport = singleResult['weatherReport']

            if (weatherReport == None):
                continue

            if (measurement == None):
                continue

            dataHour = [
                measurement['p10'],
                measurement['p25'],
                weatherReport['airPressure'],
                weatherReport['dewPoint'],
                weatherReport['foggProbability'],
                weatherReport['maxWindspeed'],
                weatherReport['precipitation'],
                weatherReport['sleetPrecipitation'],
                weatherReport['sunDuration'],
                weatherReport['sunIntensity'],
                weatherReport['temperature'],
                weatherReport['visibility'],
                weatherReport['windspeed']
            ]
            data.append(dataHour)
    return data

def trainFromSensors(number):
    i = 0
    now = time.time()
    sensorList = getSensorList()
    week = 7*24*60*60 
    timestamp = int(now-week)
    while i < number:
        sensorID = sensorList[i]
        data = getDataFromSensor(sensorID, str(timestamp))
        training_set = np.array(data)  
        if(training_set.shape[0]>60):
            sc = MinMaxScaler(feature_range = (0,1))
            training_set = sc.fit_transform(training_set)
            X,y = inAndOutput(training_set)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = loadModel("regressor.json","model.h5")
            model =furtherTraining(X_train,y_train,model)
            saveModel(model)
            print("saved")
        i = i+1
        print(i)
        
#data = getDataFromSensor(index, timestamp) 


###############################################
def makeScalerForY(training_set):
    scy = MinMaxScaler(feature_range = (0,1))
    scal = []
    for i in range(60, training_set.shape[0]-5):
        scal.append(training_set[i:i+5, 0:2])
    scal =  np.array(scal)
    scal = np.reshape(scal, (scal.shape[0],scal.shape[1]*2))
    scy.fit_transform(scal)
    return scy
    
def inAndOutput(training_set):
        X = []
        y = []
        for i in range(60, training_set.shape[0]-5):
            X.append(training_set[i-60:i, :])
            y.append(training_set[i:i+5, 0:2])
        X, y = np.array(X), np.array(y)
        # Reshaping
        y = np.reshape(y, (y.shape[0],y.shape[1]*2))
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

#X_train,y_train = prepData('../Data/trainwithp1p2.csv')
#X_test,y_test = prepData('../Data/testwithp1p2.csv')
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
    regressor.fit(X_train, y_train, epochs = 200, batch_size = 5)
    #scores = regressor.evaluate(X_test,y_test,verbose=0)
    #print(scores)
    return regressor
#regressor = myRegressor(X_train,y_train)

def loadModel(jsonpath,h5path):
    json_file = open(jsonpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5path)
    return loaded_model

def furtherTraining(X_train,y_train,regressor):
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
    regressor.fit(X_train,y_train, epochs = 200, batch_size = 5)
    return regressor

def saveModel(regressor):
    regressor_json = regressor.to_json()
    with open ("regressor.json", "w") as json_file:
        json_file.write(regressor_json)
    regressor.save_weights("model.h5")
#saveModel(regressor)

def predictionPlotter():
    now = time.time()
    sensorList = getSensorList()
    week = 7*24*60*60
    timestamp = int(now-week)
    data = []
    sensorID = sensorList[25]
    data = getDataFromSensor(sensorID, str(timestamp))
    training_set = np.array(data)
    scy = makeScalerForY(training_set)
    sc = MinMaxScaler(feature_range = (0,1))
    training_set = sc.fit_transform(training_set)
    X,y = inAndOutput(training_set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = loadModel("regressor.json","model.h5")
    prediction = model.predict(X_test)
    prediction2 = scy.inverse_transform(prediction)
    y_testU = scy.inverse_transform(y_test)
    plot2(y_testU[:,5],'real')
    plot2(prediction2[:,5],'predicted')
#predictionPlotter()

def predictionGiver(sensorID, latestTimestamp):
    urlFull = "http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/bySensor?sensor=" + str(sensorID) + "&timestamp="+str(latestTimestamp)
    print(urlFull)
    with urllib.request.urlopen(urlFull) as url:
        latestMeasurement = json.loads(url.read().decode())
        isContinuous = latestMeasurement['continuous']
        if (isContinuous != True):
            return None

    week = 7*24*60*60
    timestamp = int(latestTimestamp-week)
    data = []
    data = getDataFromSensor(sensorID, str(timestamp))
    dataArray = np.array(data)
    scy = makeScalerForY(dataArray)
    sc = MinMaxScaler(feature_range = (0,1))
    dataArray = sc.fit_transform(dataArray)
    X,y = inAndOutput(dataArray)
    model = loadModel("regressor.json","model.h5")
    prediction = model.predict(X)
    prediction2 = scy.inverse_transform(prediction)
    return prediction2[-1,:]

def plot(real,predicted):
    plt.plot(real, color = 'red', label = 'Real ')
    plt.plot(predicted, color = 'blue', label = 'Predicted')
    plt.title('P1 predic')
    plt.xlabel('Time')
    plt.ylabel('P1')
    plt.legend()
    plt.show()
#plot(y_test,predicted)

def plot2(plot,name):
    plt.plot(plot, color = 'red', label = 'graph ')
    plt.title(name)
    plt.legend()
    plt.show()
#plot2(y_testU[:,5],'real')
#plot2(prediction[:,5],'predicted')
#regressor = furtherTraining(X_test,y_test,regressor)
'''
nur mit einem datenset testen best loss = 0.01
mit zwei 0.0035
accuracy beidemal 0.2
wenn man mit einem trainiert und dem anderen tesete loss von 0.08

'''

key = os.environ.get('API_KEY')
if (key == None):
    print("No key specified, printing predictions to console only.")

sensorList = getSensorList()
latestTimestamp = int(time.time())

for i, sensorId in enumerate(sensorList):
    print(str(i) + " / " + str(len(sensorList)))
    result = predictionGiver(sensorId, latestTimestamp)
    if (type(result) != type(None)):
        if (key == None):
            print(result)
        else:
            requests.post('http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/updatePredictions',
                        json={"startTime": latestTimestamp,
                                "sensor": sensorId, 
                                "values": result.tolist(),
                                "apiKey": key})
    else:
        print(sensorId + " is not continuous!")
