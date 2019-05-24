# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:43:49 2019

@author: Josua
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

from keras.models import model_from_json

###################################################
# Part 3 - Making the predictions and visualising the results
###############load model from json
def loadModel(jsonpath,h5path):
    json_file = open(jsonpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5path)
    return loaded_model
loaded_model = loadModel('regressor.json','model.h5')
# Getting the real humdity
def predict(path,model):
    dataset_test = pd.read_csv(path)
    real = dataset_test['P1'].values
    sc = MinMaxScaler(feature_range = (0,1) )
    inputP1 =dataset_test['P1'].values
    inputP2 =dataset_test['P2'].values
    inputs = np.column_stack((inputP1,inputP2))
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(237-60,237):
        X_test.append(inputs[i,:])
    #predicted_humidity = regressor.predict(X_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
    predicted = model.predict(X_test)
    predicted_unscaled = sc.inverse_transform(predicted[:,0:2])#wir müssen slicen weil sc 2 spalten hatte jetzt aber 5
    lastrow = predicted[-1,:]
    lastrow = np.column_stack((lastrow,lastrow))
    lastrow = sc.inverse_transform(lastrow)
    predict = lastrow[:,0]
    predicted = predicted_unscaled[:,0] #die zweite spalte ist mit p2 scaliert also nutzlos
    print(predict)
    predictfromreal = np.concatenate((real,predict))# so that we have up to the point we predict + the prediction as a graph
    return predict,predictfromreal,predicted, real
predict,predictfromreal,predicted, real = predict('../Data/testwithp1p2.csv',loaded_model)
# Visualising the results
def plot(real,predicted):
    plt.plot(real, color = 'red', label = 'Real ')
    plt.plot(predicted, color = 'blue', label = 'Predicted')
    plt.title('P1 predic')
    plt.xlabel('Time')
    plt.ylabel('P1')
    plt.legend()
    plt.show()
plot(real,predictfromreal)

def plot2(plot,name):
    plt.plot(plot, color = 'red', label = 'graph ')
    plt.title(name)
    plt.legend()
    plt.show()
plot2(real,'real')
plot2(predictfromreal,'predicted')