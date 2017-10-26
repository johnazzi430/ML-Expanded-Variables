


######## Goal: Using expanded prediction parameters see if ML can yeild useful results
from __future__ import division

import pandas as pd
import numpy as np
import talib
import datetime
import scipy.stats
from pandas_datareader import Options , data , wb
import random
import pickle
import os
import csv



def context():
      return



#####                               Data Processing

def Indicator_soup(df_data):
    close = np.asarray(df_data['Close'])
    df_data['RSI'] = talib.RSI(close)
    df_data['EMA'] = talib.EMA(close, timeperiod=30)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df_data['upperband'] = upperband ; df_data['middleband'] = middleband ; df_data['lowerband']=lowerband
    MACD, MACDsig, macdhist =  talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df_data['MACD'] = MACD; df_data['MACDsig'] = MACDsig; df_data['macdhist'] = macdhist
    df_data['EMA52'] = talib.EMA(close, timeperiod=52)
    df_data['EMA20'] = talib.EMA(close, timeperiod=30)
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
    for n in df_data['Date']:
        out.append(n.weekday())
    df_data['Weekday'] = out
    return df_data['Weekday']

def Moar_indicators(df_data):


    out =[]
    for index,row in df_data.iterrows():
        row = row
        #########-----------------------------------
        against_last_2 = []
        #########-----------------------------------
        n=1
        m=1
        while n < 40:
            try:
                close_n = df_data['Close'].iloc[index-m]
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
        print(' error no regularization method set')
        raise
    return df_data

def Y_series(df_data):
    Y_series_out =[]
    if context.method == 'class':
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
    else:
        for row in df_data.iterrows():
            index, data = row
            try:
                Y_series_out.append(np.max(df_data['Close'][index+1:index+6]))
            except:
                Y_series_out.append(0)

    return Y_series_out
### -------------------------------- normalization
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
            try:
                normalized =  (da_set - np.min(da_set)) / (np.max(da_set) - np.min(da_set))
                #normalized = da_set - np.mean(da_set)
                p = [1 / float(1 + np.exp(- x)) for x in normalized ]
                df_data[column] = normalized
            except:
                pass
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
                    MEAN = np.mean(df_data[column][i-context.normlookback:i])
                    MAX = np.max(df_data[column][i-context.normlookback:i])
                    MIN = np.min(df_data[column][i-context.normlookback:i])
                    norm = ((df_data[column].iloc[i] - MEAN)/(MAX - MIN) + 1) / 2
                    out.append(norm)
            df_data[column] = out
    return df_data


def Returns(df_data):
    for column in df_data:
        if  column == 'Weekday' or (column == 'Y' and context.method == 'class'):
            pass
        else:
            dummy = []
            data = df_data[column]
            for i in range(1,len(data)):
                r = np.log(data.iloc[i-1]) - np.log(data.iloc[i])
                dummy.append(r)
            try:
                df_data_out[column] = dummy
            except:
                df_data_out = pd.DataFrame(dummy , columns = {'Close'})
    df_data_out['Weekday'] = df_data['Weekday'].iloc[1:]
    df_data_out['Y'] =df_data['Y'].iloc[1:]
    return df_data_out
##################### ------------------------------------------------------------------------ #####################

context.test_n = 500
context.lookback = 600
context.method = 'class'    ### class or reg
context.regularization = 'RSI' ### sigmoid , sofmax , standard  or, RSI
context.normlookback = 70
context.securtities = [ 'BA' , 'AAPL' , 'AMZN' , 'TSLA' , 'JPM' , 'WFC' , 'GM' , 'VRX' , 'CMG' , 'GE' , 'MSFT', 'JNJ' , 'XOM' , 'PG' , 'SBUX' , 'GS' , 'NFLX'] # Boeing
#context.securtities = [ 'BA' , 'AAPL' ,  'AMZN']
#df_data_m = Merge_dict(df_data)

end = (datetime.date.today() - datetime.timedelta(days=1))
start = end - datetime.timedelta(days=context.lookback)
df_data = {}
path = 'Data_Dump'


for tick in context.securtities:
    print('downloading', tick)
    df_data[tick] = data.DataReader(tick, 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

reform = {(outerKey, innerKey): values for outerKey, innerDict in df_data.items() for innerKey, values in
          innerDict.items()}
df_data = pd.DataFrame(reform)
pd.DataFrame.to_csv(df_data, os.path.join(path, r'historical_data.csv'))
df_data_new ={}

for tick in df_data.keys().levels[0]:
    print('stiring...', tick)
    n_df = pd.DataFrame(df_data[tick])
    context.df = n_df
    n_df = Indicator_soup(n_df)
    n_df = n_df.reset_index()
    df_data_WD = Weekday(n_df)
    n_df['Weekday'] = df_data_WD
    n_df = n_df.drop('Date', 1)
    n_df = Moar_indicators(n_df)
    context.dat = n_df
    n_df['Y'] = Y_series(n_df)
    n_df = Returns(n_df)
    n_df = n_df[:-5]  ### drop last 5
    n_df = n_df.reset_index()
    n_df = Normalize(n_df)
    df_data_new[tick] = n_df


df_data_out = pd.DataFrame(df_data_new[list(df_data_new.keys())[0]])
for tick in list(df_data_new.keys()):
    df_data_out = df_data_out.append(df_data_new[tick],ignore_index=True)

#pd.DataFrame.to_csv(df_data_out, os.path.join(path, r'historical_data_processed.csv'))

df_context = [context.test_n ,context.lookback , context.method ,context.regularization ,context.normlookback,context.securtities]
filename = os.path.join(path, r'setup_context')
pd.DataFrame.to_csv(pd.DataFrame(df_context), os.path.join(path, r'setup_context.csv'))


print('done')