# WindChaser
This is the DIRECT Project page for WindChaser group. This package is developed for wind power forecasting along with wind energy economic dispatch. 


# User Documentation

![alt text](https://github.com/yiwen26/WindChaser/blob/master/Docs/use_case.png)

## Brief Description:
        * Forecast power generation and help with usage decision

## User Input Data:
        * time horizon (MM/YY/DD, 00:00-00:00)
        * Location (zipcode)
        * User demand (Power consumption)

## Output (prediction):
        * Power generation at any time series
        * Decision on the use of renewable energy
        

## Pre_Process Module
### Corr_analysis(historyData, method)
Give an initial analysis of input data. Analyze the distribution, temporal correlation and spatio-temporal correlation.

Params:
* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.

* method - A string represents which feature is going to be analyzed with repect to input data.

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
Numpy array to time-series forecasts.

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

## Reinforcement Learning module
### Decision_maker(historyData, Predictfuture, user_profile)
Make decision on the use of wind energy based on historyData and prediction module, given a fixed user profile.

The user has the choice of either using renewable energy or using traditional energy. In order to minimize its costs, it has to choose the energy provider based on the stochastic generation profile of wind as well as its own electricity usage profile.

Params:
* historyData - A .csv input with shape of (history time length, number of samples). The values in each entry of this .csv file is the wind power generation in MWh. Along with the file we also include information such as wind site id, location, the total period of time and date.

* Predictfuture -  The neural network forecasts class which could provide accurate future forecasts. Such forecasts is used for decision-making.

* user_profile - A numpy array which includes a specific customer's one-day energy consumption with respect to time.
