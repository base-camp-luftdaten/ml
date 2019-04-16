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

dataset_train = pd.read_csv('trainingData_2019-04-09_bme280_sensor_4024.csv')
training_set = dataset_train.iloc[:, 10:11].values
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 271):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#########################################################
# Part 2 - Building the RNN

# Importing the Keras libraries and packages

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
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
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


###################################################
# Part 3 - Making the predictions and visualising the results

# Getting the real humdity
dataset_test = pd.read_csv('testData_2019-04-09_bme280_sensor_3444.csv')
real_humidity = dataset_test.iloc[:, 10:11].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['humidity'], dataset_test['humidity']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, dataset_test.shape[0]+61):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_humidity = regressor.predict(X_test)
predicted_humidity = sc.inverse_transform(predicted_humidity)

# Visualising the results
plt.plot(real_humidity, color = 'red', label = 'Real ')
plt.plot(predicted_humidity, color = 'blue', label = 'Predicted')
plt.title('humidity predic')
plt.xlabel('Time')
plt.ylabel('humidity')
plt.legend()
plt.show()
