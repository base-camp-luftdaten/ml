# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:43:25 2019
@author: Josua
"""


# Importing the libraries
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import requests
import time
import os
import socket

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense

# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import keras
import tensorflow as tf


# system specific values, as our batch size is fairly small cpu should be faster.
# change these to fit your system
# config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 128} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

# get a list with all sensors that can be used.
def getSensorList():
    response = requests.get(
        "http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/sensors", timeout=10)
    data = response.json()

    sensorList = []
    for sensor in data.keys():
        sensorList.append(sensor)

    return sensorList


'''
Starting from timestamp, retrieves all measurements
made by the sensor in 1 hour steps, up to now
'''


def getDataFromSensor(sensorID, timestamp):
    data = []
    fullData = []
    fullUrl = "http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/bySensorUntilNow?sensor=" + \
        sensorID + "&timestamp=" + str(timestamp)

    try:
        response = requests.get(fullUrl, timeout=10)
        fullData = response.json()
    except (requests.exceptions.ReadTimeout, socket.timeout, requests.exceptions.ConnectTimeout) as e:
        print("Timeout on " + fullUrl)
        print(e)
        return []

    lastWeatherReport = None
    for singleResult in fullData:
        measurement = singleResult['measurement']
        weatherReport = singleResult['weatherReport']

        if (weatherReport == None):
            # use the last available weatherReport instead
            weatherReport = lastWeatherReport
        else:
            lastWeatherReport = weatherReport

        if (measurement == None):
            return []

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

# trains our model with a given number of sensors.


def trainFromSensors(number):
    i = 100
    now = time.time()
    sensorList = getSensorList()
    week = 7*24*60*60
    timestamp = int(now-week*2)
    while i < number:
        sensorID = sensorList[i]
        data = getDataFromSensor(sensorID, str(timestamp))
        training_set = np.array(data)
        if (training_set.shape[0] > 40):
            print(sensorID)
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set = sc.fit_transform(training_set)
            X, y = inAndOutput(training_set)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            # model = loadModel("regressor.json","model.h5")
            model = myRegressor(X_train, y_train)
            # model =furtherTraining(X_train,y_train,model)
            # saveModel(model)
            print(sensorID)

        i = i+1
    return model

# data = getDataFromSensor(index, timestamp)


###############################################
# a scaler to enable the inverse scaling of the prediction
def makeScalerForY(training_set):
    scy = MinMaxScaler(feature_range=(0, 1))
    scal = []
    for i in range(40, training_set.shape[0]-5):
        scal.append(training_set[i:i+5, 0:2])
    scal = np.array(scal)
    scal = np.reshape(scal, (scal.shape[0], scal.shape[1]*2))
    scy.fit(scal)
    return scy

# reshape the data so that i can be used easier


def inAndOutput(training_set):
    X = []
    y = []
    for i in range(40, training_set.shape[0]-5):
        X.append(training_set[i-40:i, :])
        y.append(training_set[i:i+5, :2])
    X, y = np.array(X), np.array(y)
    # Reshaping
    y = np.reshape(y, (y.shape[0], y.shape[1]*2))
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    return X, y


#########################################################
# creating the model
def myRegressor(X_train, y_train):
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=30, return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=30, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=30))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units=y_train.shape[1]))
    # Compiling the RNN
    regressor.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs=100, batch_size=5)
    # scores = regressor.evaluate(X_test,y_test,verbose=0)
    # print(scores)
    return regressor
# regressor = myRegressor(X_train,y_train)
# loading the model with a path


def loadModel(jsonpath, h5path):
    json_file = open(jsonpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5path)
    return loaded_model
# loading and training on a model


def furtherTraining(X_train, y_train, regressor):
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=100, batch_size=5)
    return regressor
# saving the model


def saveModel(regressor):
    regressor_json = regressor.to_json()
    with open("regressor.json", "w") as json_file:
        json_file.write(regressor_json)
    regressor.save_weights("model.h5")

# plotting the predictions, one hour and 5 hours in the future (p10)
# also prints the error of the plotted predictions


def predictionPlotter():
    now = time.time()
    sensorList = getSensorList()
    week = 7*24*60*60
    timestamp = int(now-week*2)
    data = []
    sensorID = sensorList[34]
    data = getDataFromSensor(sensorID, str(timestamp))
    training_set = np.array(data)
    scy = makeScalerForY(training_set)
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = sc.fit_transform(training_set)
    X, y = inAndOutput(training_set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = loadModel("regressor.json", "model.h5")
    prediction = model.predict(X_test)
    prediction2 = scy.inverse_transform(prediction)
    y_testI = scy.inverse_transform(y_test)

    plot2(y_testI[:, 0], prediction2[:, 0], 'predicted2')
    plot2(y_testI[:, 8], prediction2[:, 8], 'predicted2')
    print(mean_squared_error(y_test[:, 0], prediction[:, 0]))
    print(mean_squared_error(y_test[:, 8], prediction[:, 8]))

# check if a sensor is continuous


def isContinuous(sensorID, timestamp):
    fullUrl = "http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/bySensor?sensor=" + \
        str(sensorID) + "&timestamp="+str(latestTimestamp)

    data = {}

    try:
        # the api should ideally be able to repond within 50-60ms
        # to answer whether a sensor is continuous
        response = requests.get(fullUrl, timeout=2)
        data = response.json()
    except (requests.exceptions.ReadTimeout, socket.timeout, requests.exceptions.ConnectTimeout) as e:
        print("Timeout on " + fullUrl)
        print(e)
        return False

    isContinuous = data['continuous']

    return isContinuous == True


# returns the next 5 hours of p10 and p25
def predictionGiver(sensorID, latestTimestamp):
    if isContinuous(sensorID, latestTimestamp) != True:
        return None

    week = 7*24*60*60
    timestamp = int(latestTimestamp-week)
    data = []
    data = getDataFromSensor(sensorID, str(timestamp))
    if (data == []):
        return None
    dataArray = np.array(data)
    scy = makeScalerForY(dataArray)
    sc = MinMaxScaler(feature_range=(0, 1))
    dataArray = sc.fit_transform(dataArray)
    X, y = inAndOutput(dataArray)
    model = loadModel("regressor.json", "model.h5")
    prediction = model.predict(X)
    prediction2 = scy.inverse_transform(prediction)
    return prediction2[-1, :]

# unused function that could create a plot


def plot(real, predicted):
    plt.plot(real, color='red', label='Real ')
    plt.plot(predicted, color='blue', label='Predicted')
    plt.title('P1 predic')
    plt.xlabel('Time')
    plt.ylabel('P1')
    plt.legend()
    plt.show()


# function used in predictionPlotter() to create the plots
def plot2(plot, plot2, name):
    plt.plot(plot, color='red', label='real ')
    plt.plot(plot2, color='blue', label='prediction ')
    plt.title(name)
    plt.legend()
    plt.show()


# giving the prediction to the backend
key = os.environ.get('API_KEY')
if (key == None):
    print("No key specified, printing predictions to console only.")

sensorList = getSensorList()
latestTimestamp = int(time.time())

for i, sensorId in enumerate(sensorList):
    # check if process has been running for more than 50 minutes (= 3000 sec)
    if (int(time.time()) - latestTimestamp > 3000):
        print("Stopping execution because process took too long.")
        break

    print(str(i) + " / " + str(len(sensorList)))
    result = predictionGiver(sensorId, latestTimestamp)
    if (type(result) != type(None)):
        if (key == None):
            print(result)
        else:
            try:
                requests.post('http://basecamp-demos.informatik.uni-hamburg.de:8080/AirDataBackendService/api/measurements/updatePredictions',
                              timeout=5,
                              json={"startTime": latestTimestamp,
                                    "sensor": sensorId,
                                    "values": result.tolist(),
                                    "apiKey": key})
            except (requests.exceptions.ReadTimeout, socket.timeout, requests.exceptions.ConnectTimeout) as e:
                print("Prediction upload took too long " +
                      sensorId + " " + str(latestTimestamp))
                print(e)
    else:
        print(sensorId + " is not continuous!")
