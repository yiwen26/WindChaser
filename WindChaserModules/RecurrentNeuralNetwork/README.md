# Code for the Recurrent Neural Network (Without Wind Power Input)

This code based on Tensorflow. Please make sure you have downloaded the package if you want to use RNN_5input_XXJ.py

## Install TensorFlow

```
$ pip install tensorflow      # Python 2.7; CPU support
```

## Usage:

```
$ python RNN_5input_XXJ.py
```





## FAQ
### 1) Why we chose a time step of 6 (which means 6 hours with interval of 1hour)?

![Alt](https://github.com/yiwen26/WindChaser/blob/master/Graphs/RMSE%20Progression%20of%20LSTM%20vs.%20Time%20Steps.png)

After we built the LSTM model, we trained it with different time steps(from 6 to 18), and recorded the root-mean-square errors, as the above figure show, our LSTM model reported a lowest prediction error at the time step of 6.
