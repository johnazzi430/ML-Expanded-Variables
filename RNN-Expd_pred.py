from __future__ import division

import numpy as np
import theano
import keras
import os
import pickle
import pandas as pd
import random


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


def prepareXandY_binary(dataset):
    context.end = len(dataset['Close'])
    exclude = random.sample(list(range(0, context.end - 1)), context.test_n)

    X_learn = []
    Y_learn = []
    X_test = []
    Y_test = []

    for row in dataset.iterrows():
        index, data = row

        if any(N == True for N in pd.isnull(data.tolist())):
            continue
        elif any(N == 0 for N in data.tolist()):
            continue
        else:
            Y = data.tolist()[-1]
            if Y > 0 :
                Y2=[1,0]
            else:
                Y2=[0,1]

            X = data.tolist()[:-1]

            if any(z == index for z in exclude) == False:
                X_learn.append(X)  ## trying to predict total return, total return y is in position -2
                Y_learn.append(Y2)
            else:
                X_test.append(X)
                Y_test.append(Y2)

    learn_len = len(X_learn)
    test_len = len(X_test)
    if (learn_len % 2 == 0 ):
        context.X_learn = np.asarray(X_learn)
        context.Y_learn = np.asarray(Y_learn)
    else:
        context.X_learn = np.asarray(X_learn[0:learn_len-1])
        context.Y_learn = np.asarray(Y_learn[0:learn_len-1])

    if (test_len % 2 == 0 ):
        context.X_test = np.asarray(X_test)
        context.Y_test = np.asarray(Y_test)
    else:
        context.X_test = np.asarray(X_test[0:test_len-1])
        context.Y_test = np.asarray(Y_test[0:test_len-1])

    context.exclude = exclude
    return

                ### ----------------- starting parameters ------------------------- ##

def context():
    return

def Y_index(data):
    Y_out = []
    for Y in data[:,0]:
        if Y > .5:
            Y_out.append(1)
        if Y < .5:
            Y_out.append(-1)
    return Y_out

path ='C:\\Users\\John\Desktop\\Investing\\Machine Learning\Data Dump'

context.data_stored = 'setup_context.csv'  # stored or new
context.data_stored_processed = True #or False

filename = os.path.join(path, context.data_stored)
df_cont = pd.read_csv(filename)
context.test_n = int(df_cont.iloc[1][1])
context.lookback = int(df_cont.iloc[0][1])
context.method = df_cont.iloc[2][1]
context.regularization = df_cont.iloc[3][1]
context.normlookback = df_cont.iloc[4][1]
context.securtities = df_cont.iloc[5][1]

df_data = pd.read_csv(os.path.join(path, r'historical_data_processed.csv'),header=0)
prepareXandY_binary(df_data)

x_train = context.X_learn
y_train = context.Y_learn
x_test = context.X_test
y_test = context.Y_test


up = sum([1 for x in y_test if x[0] > 0])
down = sum([1 for x in y_test if x[1] > 0])
check = up / (up + down)

batch_size = 10
variables = x_train.shape[1]

x_train2 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test2 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential()
model.add(LSTM(400, input_shape=(x_train2.shape[1], x_train2.shape[2])))
model.add(Dense(100, activation='softmax'))#,input_shape=(x_train2.shape[1], x_train2.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(400, activation='softmax'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train2, y_train, epochs=10, batch_size=batch_size)


trainPredict = model.predict(x_train2, batch_size=batch_size)
trainP_out = Y_index(trainPredict)

testPredict = model.predict(x_test2, batch_size=batch_size)

