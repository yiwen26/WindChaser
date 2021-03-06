# Recurrent Neural Network (Without Wind Power Input)

This code based on ``Tensorflow``. Please make sure you have already downloaded the package if you want to use ``RNN_5input_XXJ.py`` and ``RNN_5input_XXJ.ipynb``

## Install TensorFlow
```
$ pip install tensorflow      # Python 2.7; CPU support
```

## Usage:
Use the command ``nosetests `` if you want to run the unit tests in terminal, the results of unit tests shows as
```
----------------------------------------------------------------------
Ran 3 tests in 38.286s

OK
```

If you want to run the ``RNN_5input_XXJ.ipynb`` to see the training and predication results, please make sure to clear all the output first, in order to initialize all the weights and bias in LSTM.


## Some Methods Explaination: 

* MinMaxScaler()
It can rescale the data to the range of [0, 1], also called normalizing.
Before building the RRN, we normalized the dataset using the MinMaxScaler() preprocessing class from the scikit-learn library first. Because LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used. 

* min_max_scaler.transform(X_test)
The same instance of the transformer can then be applied to some new test data unseen during the fit call: the same scaling and shifting operations will be applied to be consistent with the transformation performed on the train data.

* cell.zero_state(batch_size, tf.float32)
At each time step, it will reinitialising the hidden state, return a N-D tensor of shape [batch_size, state_size] filled with zeros.

* Want to see more...
  Find detailed documentation in the ``RNN_5input_XXJ.ipynb`` !


## Training and Predication Results:
We set the training parameters as: batch size of 14, hidden layer number is 10, time steps is 12, and after the model being trained 100 times, the comparision of true values and forecasted values of testing data are ploted as below: 

![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/Wind%20Power%20Forecasting%20(Without%20history%20power%20values).png)

|        Mean absolute error |  Root mean squared error  | 
|---------------------------:|:-------------------------:|
|                      0.138 |                     0.216 | 

* The red line represents the true testing values of wind power, the blue line is our predicated values
* we can see our model shows a very accurate forcasting. This maybe because that our input data includes alomst all the weather feather (including the wind direction, wind speed, air temperature, surface air pressure, and density at hub height) that may influence the wind power. 
* Compared with the other model, which included the history wind power values as input, this one shows a slightly lower predication accuracy.

## FAQ
### 1) Why we chose a time step of 12 (which means 12 hours with interval of 1 hour)?

<img src="https://github.com/yiwen26/WindChaser/blob/master/Graphs/RMSE_vs_TimeSteps.png" width="480">

After we built the LSTM model, we trained it with batch size of 14, RNN unit = 10, and time steps from 2 to 18, and recorded the root-mean-square errors, as the above figure show, our LSTM model reported a lowest prediction error at the time step of 12.


### 2) How we chose the rnn_unit and batch size?

![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/Training_Parameters.png)

We traind our LSTM model and recorded the RMSE values and training time, as shown in the above list, if we use a smaller hidden layer number, say RNN unit = 5, our predication error will increase. As for the batch size, using a small batch size (eg. batch size =2), all though our model will report a higher accuracy, the training time is too long; while the LSTM model with a large batch size yields a lower prediction accuracy. Therefore, we decided to use a hiden layer number of 10 and a batch size of 14 to train our data set, which gives a adequate accuracy as well as a reasonable training time. 
