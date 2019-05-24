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


from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

###############################################
def prepData(filepath):
    dataset_train = pd.read_csv(filepath)
    training_set = np.column_stack((dataset_train['P1'],dataset_train['P2']))
    # Feature Scaling
    
    sc = MinMaxScaler(feature_range = (0,1) )
    scaled_set = sc.fit_transform(training_set)
    
    # Creating a data structure with 60 timesteps and 1 output
    def inAndOutput(training_set):
        X_train = []
        y_train = []
        for i in range(60, training_set.shape[0]-5):
            X_train.append(training_set[i-60:i, :])
            y_train.append(training_set[i:i+5, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        # Reshaping
        y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1]))
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        return X_train,y_train
    X_train,y_train = inAndOutput(scaled_set)
    return X_train,y_train

X_train,y_train = prepData('../Data/trainwithp1p2.csv')

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
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    #regressor.fit_generator((generator=training_generator,
            #        validation_data=validation_generator,
             #       use_multiprocessing=True,
              #      workers=6)
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