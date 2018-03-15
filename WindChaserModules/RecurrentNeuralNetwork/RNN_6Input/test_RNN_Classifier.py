import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import RNN_Classifier
import os




def test_get_data():
    batch_index,train_x,train_y,test_x,test_y,train_scaler=RNN_Classifier.get_data()
    assert np.array(train_x).shape==(5994, 6, 6),"you are wrong!!idiot!!"
    assert np.array(train_y).shape==(5994, 6, 1),"you are wrong!!idiot!!"
    assert np.array(test_x).shape==(30, 6, 6),"you are wrong!!idiot!!"
    assert len(test_y)==180,"you are wrong!!idiot!!"
    return



def test_train_lstm():
    test_predict, loss_data, test_y=RNN_Classifier.train_lstm()
    assert len(loss_data)==20,"the training steps is wrong!"
    assert loss_data[0]>loss_data[-1],"The loss function is wrong!"
    assert test_y.shape==(180,1)
    assert test_predict.shape==(180,1)
                                                        
    return
