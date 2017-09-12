


######## Goal: Using expanded prediction parameters see if ML can yeild useful results
from __future__ import division

import pandas as pd
import numpy as np
import talib
import datetime
from pandas_datareader import Options , data , wb
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import pickle
import matplotlib.pyplot as plt

def context():
      return


##### Data Processing

def Indicator_soup(df_data):
    close = np.asarray(df_data['Close'])
    df_data['RSI'] = talib.RSI(close)
    df_data['EMA'] = talib.EMA(close, timeperiod=30)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df_data['upperband'] = upperband ; df_data['middleband'] = middleband ; df_data['lowerband']=lowerband
    MACD, MACDsig, macdhist =  talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df_data['MACD'] = MACD; df_data['MACDsig'] = MACDsig; df_data['macdhist'] = macdhist

    df_data['EMA_delta'] = df_data['Close'] - df_data['EMA']
    df_data['upperband_delta'] = df_data['Close'] - df_data['upperband']
    df_data['middleband_delta'] = df_data['Close'] - df_data['middleband']
    df_data['lowerband_delta'] = df_data['Close'] - df_data['lowerband']

    #close_2=np.append(close,0)
    #close_2=np.delete(close_2,close[0])
    #df_data['Close tomorrow'] = close_2
    return df_data

def Weekday(df_data):
    out =[]
    context.df = df_data['Date']
    for n in df_data['Date']:
        out.append(n.weekday())
    df_data['Weekday'] = out
    return df_data['Weekday']

def Moar_indicators(df_data):


    out =[]
    for index,row in df_data.iterrows():

        row = row

        try:
            against_last = row - df_data.iloc[index-1]
        except:
            against_last = row
        #########-----------------------------------
        against_last_2 = []
        for n in against_last:
            if n > 0:
                against_last_2.append(1)
            else:
                against_last_2.append(-1)

        #########-----------------------------------
        n=1
        m=1
        while n < 20:
            try:
                close_n = df_data['Close'].iloc[index] - df_data['Close'].iloc[index-m]
                if pd.isnull(close_n) == False:
                    against_last_2.append(close_n)
                    n = n + 1
                    m = m + 1
                else:
                    m = m+1
            except:
                m = m + 1

        #########-----------------------------------
        out.append(against_last_2)

    df_data_2 = pd.DataFrame(data = out)# , columns= df_data.keys())
    context.out = df_data_2
    context.out2 = df_data
    df_data =  pd.concat([df_data, df_data_2], axis=1)
    return df_data

def Normalize(df_data):
    if context.regularization == 'softmax':
        df_data = Normalize_SM(df_data)
    elif context.regularization == 'sigmoid':
        df_data = Normalize_Sig(df_data)
    elif context.regularization == 'standard':
        df_data = Normalize_Std(df_data)
    elif context.regularization == 'RSI':
        df_data = Normalize_RSI(df_data)
    else:
        print ' error no regularization method set'
        raise
    return df_data

### normal normalization
def Normalize_Std(df_data):
    for column in df_data:
        if column == 'Y' :
            pass
        else:
            Ave = np.mean(df_data[column])
            STD = np.std(df_data[column])
            norm =  (df_data[column] - Ave )/ STD
            norms =[]
            for n in norm:
                if pd.isnull(n) == False:
                    norms.append(n)
                else:
                    norms.append(0)
            df_data[column] = norms
    return df_data

### softmax
def Normalize_SM(df_data):   ##### Softmax
    for column in df_data:
        if column == 'Weekday':
            pass
        else:
            da_set = df_data[column]
            Y =[]
            for n in da_set:
                if pd.isnull(n) == False:
                    Y.append(n)
                else:
                    Y.append(0)
            y = np.atleast_2d(Y)
            axis = 1
            y = y - np.expand_dims(np.max(y, axis=axis), axis)
            y = np.exp(y)
            ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
            p = y / ax_sum
            context.p = p
            df_data[column] = p[0]
    return df_data    # softmax normilization

#sigmoid
def Normalize_Sig(df_data):   ##### sigmoid
    for column in df_data:
        if column == 'Weekday':
            pass
        else:
            da_set = df_data[column]
            p = [1 / float(1 + np.exp(- x)) for x in da_set]
            df_data[column] = p
    return df_data    # sigmoid

#RSI
#The idea here will be to call a relative strength index since we are working with timeseries data
#standard normalization may not be value added
def Normalize_RSI(df_data):
    for column in df_data:
        if  column == 'Weekday' or (column == 'Y' and context.method == 'class'):
            pass
        else:
            out =[]
            for i , d in df_data[column].iteritems():
                if i <= context.normlookback :
                    out.append(0)
                else:
                    MEAN = np.mean(df_data[column][i-context.normlookback:i+1])
                    MAX = np.max(df_data[column][i-context.normlookback:i+1])
                    MIN = np.min(df_data[column][i-context.normlookback:i+1])
                    norm = ((df_data[column].iloc[i] - MEAN)/(MAX - MIN) + 1) / 2
                    out.append(norm)
            df_data[column] = out
    return df_data

### Define Y
def Y_series(df_data):
    Y_series_out =[]
    for row in df_data.iterrows():
        index, data = row
        try:
            close_yesterday= df_data['Close'][index]
            next_5_days_closes = df_data['Close'][index+1:index+6]
            next_5_days_closes.tolist()[4]
            if any( next_5_days_closes > close_yesterday):
                Y_series_out.append(1)
            else:
                Y_series_out.append(-1)
        except:
            Y_series_out.append(0)

    return Y_series_out

##### Prepare for ML
def prepareXandY(dataset):
    context.end = len(dataset['Close'])
    exclude = random.sample(xrange(0, context.end - 1), context.test_n)

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
            X = data.tolist()[:-1]

            if any(z == index for z in exclude) == False:
                X_learn.append(X)  ## trying to predict total return, total return y is in position -2
                Y_learn.append(Y)
            else:
                X_test.append(X)
                Y_test.append(Y)

    context.X_learn = X_learn
    context.Y_learn = Y_learn
    context.X_test = X_test
    context.Y_test = Y_test
    context.exclude = exclude
    return

                ### ----------------- starting parameters ------------------------- ##

######################################################################

def data_process():
    end = (datetime.date.today() - datetime.timedelta(days=1))
    start = end - datetime.timedelta(days=context.lookback)
    frames = []
    for tick in context.securtities:
        print 'processing' , tick
        df_data = data.DataReader(tick, 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            ##
        df_data = Indicator_soup(df_data)
        df_data = df_data.reset_index()
        df_data_WD = Weekday(df_data)
        df_data = df_data.drop('Date', 1)
        df_data = Moar_indicators(df_data)
        df_data['Weekday'] =  df_data_WD
        df_data['Y'] = Y_series(df_data)
        df_data = df_data[:-5]              ### drop last 5
        df_data = Normalize(df_data)
            ##
        frames.append(df_data)
    df_data = pd.concat(frames)
    return df_data

    ##################

##################### ------------------------------------------------------------------------ #####################
context.test_n = 20
context.lookback = 500
context.method = 'class'    ### class or reg
context.regularization = 'RSI'
context.normlookback = 20
context.securtities = [ 'BA' , 'AAPL' , 'AMZN' , 'TSLA' , 'JPM' , 'WFC' , 'GM' , 'VRX' , 'CMG' , 'GE' , 'MSFT', 'JNJ' , 'XOM' , 'PG' , 'SBUX' , 'GS' , 'NFLX'] # Boeing
context.securtities = [ 'BA' , 'AAPL' ]
df_data = data_process()
prepareXandY(df_data)




##################### ------------------------------------------------------------------------ #####################


if context.method == 'class':
    context.clf = MLPClassifier(hidden_layer_sizes=(60, 60 , 10), activation='logistic' , solver= 'sgd',
                               max_iter=2000, alpha=1, verbose= False, tol=1e-7, learning_rate= 'invscaling',
                                learning_rate_init=1, shuffle=True, warm_start=True, early_stopping=False , random_state=1)
elif context.method == 'reg':
    context.clf = MLPRegressor(hidden_layer_sizes=(40, 40 , 10 , 10 , 10), activation='tanh' , solver= 'sgd',
                               max_iter=2000, alpha=0.0001, verbose= True, tol=1e-4, learning_rate= 'invscaling',
                                learning_rate_init=0.01, shuffle=True, warm_start=True, early_stopping=False)
elif context.method == 'bayes':
    context.clf =
i=0
accuracy_last = 0
iterations = 0
while i < 5000:
    rannnd = np.random.random_integers(-5,5,2).tolist()
    rand2 = np.random.random_integers(0,5000).tolist()
    context.clf = MLPClassifier(hidden_layer_sizes=(60, 60, 10), activation='logistic', solver='sgd',
                                max_iter=200, alpha=10**rannnd[0], verbose=False, tol=1e-7, learning_rate='invscaling',
                                learning_rate_init=10**rannnd[1], shuffle=True, warm_start=True, early_stopping=False,
                                random_state=rand2)
    for n in range(0,len(context.Y_learn),100):
        B = context.clf.partial_fit(context.X_learn[n:n+100] , context.Y_learn[n:n+100] , classes=np.unique(context.Y_learn))
        accuracy = context.clf.score(context.X_test, context.Y_test)
        print 'Loss', context.clf.loss_ , '/ Accuracy' , round(accuracy,4) , '/ Iter',iterations
        if accuracy > accuracy_last:
            context.accuracy_best = accuracy
            context.clf_best = B
        accuracy_last = accuracy
        iterations = iterations +1
    i=i+1
print context.accuracy_best
up = sum([1 for x in context.Y_test if x >= 0])
down = sum([1 for x in context.Y_test if x < 0])
check = up / (up + down)
context.clf = context.clf_best
#context.clf.fit(context.X_learn, context.Y_learn)
accuracy = context.clf.score(context.X_test, context.Y_test)
print accuracy
print check
#prob = context.clf.predict_proba(df_data.iloc[-1][:-1])

filename = 'ML_exp_predic.sav'
pickle.dump(context.clf, open(filename, 'wb'))

#######         Check
Aout = []
Bout = []
for x in range(1,900):
        randomnum = np.random.random_integers(1,len(context.Y_test))
        A = context.clf.predict(np.asarray(df_data.iloc[-randomnum][:-1]).reshape(1, -1))
        B = context.Y_test[-randomnum ]
        Aout.append(A[0])
        Bout.append(B)
A = Aout
A2 = ((A-np.mean(A)) / ( np.max(A) - np.min(A)) +1 ) / 2
B2 = np.asarray(Bout)

d = [A2 , B2]
d.transpose()
df_out = pd.DataFrame( data = np.transpose(d) , columns= [ 'percentile' , 'Yout'])

df_out_up = df_out.mask(df_out['Yout'] < 0)
df_out_down = df_out.mask(df_out['Yout'] > 0)

up_mean = np.mean(df_out_up['percentile'])
up_std = np.std(df_out_up['percentile'])
down_mean = np.mean(df_out_down['percentile'])
down_std = np.std(df_out_down['percentile'])

[up_mean , up_std , down_mean , down_std]                    # %%% what we are doing here is normalizing predicted valyes. showing that in deed we have a accuracy

if np.std(Aout) > 0.001 :
    print 'unsuccessful'
elif np.std(Aout)> 0.001:
    print 'well its different'


#context.clf.predict(df_data.iloc[-1][:-1])
#context.clf.predict_proba(df_data.iloc[-1][:-1])

