# Code for the Recurrent Neural Network (Without Wind Power Input)

This code based on Tensorflow. Please make sure you have downloaded the package if you want to use ``RNN_5input_XXJ.py``

## Install TensorFlow

```
$ pip install tensorflow      # Python 2.7; CPU support
```

## Usage:
Use the following command if you want to run the ``RNN_5input_XXJ.py`` in terminal
```
$ python RNN_5input_XXJ.py
```
If you want to run the ``RNN_5input.ipynb``, please make sure to clear all the output first, in order to initialize all the weights and bias in LSTM.

## Training and Predication Results:

![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/Wind%20Power%20Forecasting%20(Without%20history%20power%20values%20input).png)

The red line represents the true testing values of wind power, the blue line is our predicated values, we can see our model shows a very accurate forcasting. This maybe because that our input data includes alomst all the weather feather (including the wind direction, wind speed, air temperature, surface air pressure, and density at hub height) that may influence the wind power. 


## FAQ
### 1) Why we chose a time step of 6 (which means 6 hours with interval of 1hour)?


![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/RMSE%20Progression%20of%20LSTM%20vs.%20Time%20Steps.png=80x50)

After we built the LSTM model, we trained it with different time steps(from 6 to 18), and recorded the root-mean-square errors, as the above figure show, our LSTM model reported a lowest prediction error at the time step of 6.


### 1) Why we chose the rnn_unit and batch size?
![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/TrainingParameters.png)
We traind our LSTM model and recorded the RMSE values and training time, as shown in the above list, if we use a smaller hidden layer number, say RNN unit = 5, our predication error will increase. As for the batch size, using a small batch size (eg. batch size =2), all though our model will report a higher accuracy, the training time is too long; while the LSTM model with a large batch size yields a lower prediction accuracy. Therefore, we decided to use a hiden layer number of 10 and a batch size of 14 to train our data set, which gives a adequate accuracy as well as a reasonable training time. 
