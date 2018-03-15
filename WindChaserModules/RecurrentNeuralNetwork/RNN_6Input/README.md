# Recurrent Neural Network (With Wind Power Input)

This code based on `Tensorflow`. Please make sure you have downloaded the package if you want to run `RNN_6input.ipynb`

## Install TensorFlow
```
$ pip3 install --upgrade tensorflow    #for windows 
```

## Notes:

* Everytime you want to run the ``RNN_6input.ipynb``, please make sure to clear all the past output first, in order to initialize all the weights and bias in LSTM. For convenience, you can use `kernel > restart&clear output`in menu of jupyter notebook to quickly restart.

* Use the following command if you want to test the `RNN_Classifier.py`
```
$ nosetests test_RNN_Classifier.py
```

## Training and Predication Results:

![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/Wind%20Power%20Forecasting%20(With%20history%20power%20values%20input).png)


|        Mean absolute error |  Root mean squared error  | 
|---------------------------:|:-------------------------:|
|                       0.14 |                      0.21 | 

The red line represents the true testing values of wind power, the blue line is our predicated values, our model shows the wind forecasting in high accuracy.



## Results of unit tests
```
----------------------------------------------------------------------
Ran 2 tests in 17.350s

OK

```
