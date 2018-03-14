# API User Documentation
This is the DIRECT Project page for WindChaser group. This package is developed for wind power forecasting along with wind energy economic dispatch. 

![alt text](https://github.com/yiwen26/WindChaser/blob/master/Docs/use_case.png)

## Brief Description:
* Real-time Forecasts for power generation by deep recurrent neural networks

* Online decision making for electricity users for energy costs saving

## User Input Data:
* time horizon (start time till end time)
* Location (wind site longitude and latitude)
* User demand profile (Power consumption in normalized units)

## Forecasts System Input Data 
All data has interval of 5mins, and necessary data normalization and 0-1 scaling are implemented before feeding data into machine learning algorithm
* Wind Direction (Degree, numpy 2D array)

* Wind Speed (m/s, numpy 2D array)

* Air temperature (K, numpy 2D array)

* Surface air pressue (Pa, numpy 2D array)

* density at hub height (kg/m^3, numpy 2D array)

* History wind power (MWh, numpy 2D array)

## Power Forecastrs Output (prediction):
* Power generation at any time series
* Decision on the use of renewable energy
        

## Pre_Process Module
### Corr_analysis(historyData, method, history)
Give an initial analysis of input data. Analyze the distribution, temporal correlation and spatio-temporal correlation.

Params:
* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.

* method - A string represents which feature is going to be analyzed with repect to input data.

* The length of memory taken into consideration.

We currently design three use cases for corr_analysis, which are (a). power generation distribution fitting (e.g., to a Gaussian or a Weibull distribution), (b). temporal correlation (e.g., the wind power generation auto-correlation), and (c). the spatio-temporal correlation (e.g., the Pearson correlation coefficient matrix) which takes into consideration of a group of wind farms.

Returns: 
Numpy array of info for distribution and correlation matrix.

Raises:
* ValueError - If input data is not .csv file
* ValueError - If method is out of the given selection.



## Neural Network module

### Predictfuture(historyData, pred_length, vali)
Predict the future power generation with length of *pred_length* giben historyData input.

This modules makes use of temporal learning model, e.g., Recurrent Neural Networks (RNN) and Long Short Term Memory (LSTM) to find the forecasts of future power generation based on historical observations.

Params:
* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.

* pred_length - A numpy array which defines the length of prediction horizon.

* vali - A Ture/False logic input to decide whether split training data to validation dataset.

Returns:

Numpy array of time-series forecasts.

Raises:
* ValueError - If input data is not .csv file.
* ValueError - If method is out of the given selection.
* ValueError - If the pred_length is out of a given constraint.


### Predictclass(historyData, vali)
Predict the class of generation profile current historyData lies in.

This modules makes use of neural network classifier model, e.g., convolutional neural networks (CNN) to find the output class of input time-seires.

Params:
* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.


* vali - A Ture/False logic input to decide whether split training data to validation dataset.

Returns:

Numpy array to indicate the class of history data samples, e.g., high wind, mild days or extreme days.

Raises:
* ValueError - If input data is not .csv file.
* ValueError - If method is out of the given selection.

## GUI module

## Reinforcement Learning module
### Main function

Params:

* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.

* user_profile - A numpy array which includes a specific customer's one-day energy consumption with respect to time.

### Sample_next_action(availableaction, currentstate, probability)
Make decision on the use of wind energy based on historyData and prediction module, given a fixed user profile.

The user has the choice of either using renewable energy or not using electricity energy during each time slot. In order to minimize its costs, it has to choose the energy provider based on the stochastic generation profile of wind as well as its own electricity usage profile.

Params:

availableaction

currentstate

probability

Returns:

Numpy array for hour-ahead electricity usage suggestion. Currently only returns binary decision, but can be entended to decision level in the future with consideration of multi agent. 
