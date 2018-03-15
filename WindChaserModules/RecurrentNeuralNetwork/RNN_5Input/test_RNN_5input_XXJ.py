import RNN_5input_XXJ
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import shape
import matplotlib.ticker as ticker


"""
Load data into pandas
"""
df = pd.read_csv('WindPower2012.csv', error_bad_lines=False)
print("Length of original data : ", len(df))

# calculat the mean value for every hour and save as a new dataframe
data2 =[]   
for i in range (len(df)//12):
    data2.append(df[i * 12:(i + 1) * 12].mean())
data2 = pd.DataFrame(data2)
print("Length of hourly averaged data : ", len(data2))

# Get the values of the 6-11 columns
data= data2.iloc[:, 5:11].values



"""
Set Parameters:
Next we set the RNN model parameters. We will run the data through 20 epochs, in batch sizes of 14.
The RNN will be of size 10 units.   
"""
rnn_unit = 10       #hidden layer units
input_size = 5
output_size=1
lr=0.0006         # learning rate

batch_size = 14   
time_step = 6     

train_begin=0 
train_end = 6000
test_begin = 6000
test_len = 180
iter_time = 20

# RNN output node weights and biases
weights = {
           'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
           'out':tf.Variable(tf.random_normal([rnn_unit,1]))
           }

biases = {
          'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
          'out':tf.Variable(tf.constant(0.1,shape=[1,]))
          }


# Get train data function: load training data for LSTM
# Input: batch_size, time_step, train_begin, train_end
# Output: batch_index, train_x, train_y
def test_get_train_data():  
    batch_index, train_x, train_y = RNN_5input_XXJ.get_train_data(batch_size, time_step, train_begin, train_end)
    assert shape(train_x) == ( ((train_end - train_begin)- time_step), time_step, input_size), "Function load_data not works properly!"
    assert shape(train_y) == ( ((train_end - train_begin)- time_step), time_step, output_size), "Training output in the wrong shape!" 
    assert len(batch_index) == 430, "Incorrect Number of batches!"
    print("Successfully pass test_get_train_data()")


# Get test data function: load testing data for LSTM 
# Input: time_step, test_begin, test_len
# Output: test_x, test_y, scaler_for_x, scaler_for_y
def test_get_test_data():  
    test_x, test_y, scaler_for_x, scaler_for_y = RNN_5input_XXJ.get_test_data(time_step, test_begin, test_len)
    assert shape(test_x) == ( (test_len//time_step), time_step, input_size), "Testing output in the wrong shape!"
    assert shape(test_y) == (test_len,), "Testing output in the wrong shape!"
    print("Successfully pass test_get_test_data()")


def test_train_lstm():
    test_y, test_predict, loss_list, rmse, mae = RNN_5input_XXJ.train_lstm(batch_size, time_step, train_begin, train_end, test_begin, iter_time, test_len)
    assert shape(test_predict) == shape(test_y), "True and predicated testing output shapes Do not Match!"
    assert loss_list[0] > loss_list[-1], "Loss minimization process breaks during training LSTM! "
    print("Successfully pass test_train_lstm()")
