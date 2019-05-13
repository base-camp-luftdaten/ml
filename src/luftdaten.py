# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
###############################################

dataset_train = pd.read_csv('../Data/trainwithp1p2.csv')
training_set = np.column_stack((dataset_train['P1'],dataset_train['P2']))
# Feature Scaling
def scaler(Dataset):
    sc = MinMaxScaler(feature_range = (0,1) )
    return sc
sc = scaler(training_set)
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
    return regressor
regressor = myRegressor(X_train,y_train)

###################################################
# Part 3 - Making the predictions and visualising the results

# Getting the real humdity
dataset_test = pd.read_csv('../Data/testwithp1p2.csv')
real = dataset_test['P1'].values

# Getting the predicted stock price of 2017
dataset_P1 = pd.concat((dataset_train['P1'], dataset_test['P1']), axis = 0)
dataset_P2 = pd.concat((dataset_train['P2'], dataset_test['P2']), axis = 0)
inputs_P1 = dataset_P1[len(dataset_P1) - len(dataset_test) - 60:].values
inputs_P2 = dataset_P2[len(dataset_P2) - len(dataset_test) - 60:].values
inputs_P1 = inputs_P1.reshape(-1,1)
inputs_P2 = inputs_P2.reshape(-1,1)
inputs = np.column_stack((dataset_P1,dataset_P2))
sc2 = scaler(inputs)
inputs = sc2.fit_transform(inputs)
X_test = []
for i in range(60, dataset_test.shape[0]+61):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
predicted_humidity = regressor.predict(X_test)
predicted_humidity = np.column_stack((predicted_humidity,predicted_humidity)) # das ist fürs rückskalieren
#predicted_humidity = sc2.inverse_transform(predicted_humidity[:,6:8])
predicted_humidity_unscaled = sc2.inverse_transform(predicted_humidity) #hier war problem weigen ungleichen 
#vor und nach = irgendwie noch anschauen und gewollte ergebnise rausholen
#predicted_humidity = predicted_humidity[:,0]


# Visualising the results
def plot(real,predicted):
    plt.plot(real, color = 'red', label = 'Real ')
    plt.plot(predicted, color = 'blue', label = 'Predicted')
    plt.title('P1 predic')
    plt.xlabel('Time')
    plt.ylabel('P1')
    plt.legend()
    plt.show()
plot(real,predicted_humidity)