# ml

Our Neural network is created in train.py, where it is trained and used to predict as well.

The training happens with the function trainFromSensors(number) where we continuesly train our network on multiple sensors untill we trained on the given number of sensors. This function saves and loads a model from the model.h5 and the regressor.json files.

To predict the next 5 hours the function predictionGiver can be used. this function needs a sensorID and a timestamp to know for which sensor the prediction should be made.

Apart from these core functions we also have some functions to plot the predictions so that is easier to analyse them, to save and load the modell and to prepare the training data. The function myRegressor was originnally used to create the model.

apart from the train.py file there are 2 models saved one is the one used for the prediction the other one was used to predict wheather values so that we could compare the results to the prediction of p10 and p25

we also have two graphs that visualize the results.

## Running the script

To run the script, change into the `src` directory.

In order to upload the predictions to the server, the script
should be run after setting the API Key as specified in the [backend
description](https://github.com/base-camp-luftdaten/data#installation--setup) (look for `apiKey`) as an environment variable.

For example:

```sh
API_KEY=my-api-key python3 train.py
```

Otherwise, the server is not going to accept the heatmaps/predictions.
