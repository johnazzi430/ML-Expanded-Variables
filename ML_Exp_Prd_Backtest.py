from __future__ import division
import pandas as pd
import numpy as np
import talib
import datetime
import random
import pickle

from sklearn.neural_network import MLPClassifier

from pandas_datareader import Options , data , wb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import math



#### Backtesting Module



#trading starting parameters:

def context():
      return

            ### -------------------------------trade parameters-------------------------#


context.tick = 'CMG'
context.backtest_len= 40
context.method = 'class'    ### class or reg
context.regularization = 'softmax'   ### softmax , sigmoid or standard

                        ##------------------------------------------------##
wkday = datetime.date.today().weekday()
if wkday < 5:
    end = (datetime.date.today() - datetime.timedelta(days=1))
else:
    if wkday == 5:
        end = (datetime.date.today() - datetime.timedelta(days=2))
    elif wkday == 6:
        end = (datetime.date.today() - datetime.timedelta(days=3))

start = end - datetime.timedelta(days=context.backtest_len)
df_data = data.DataReader(context.tick, 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))


########################################
file = open('ML_exp_predic.sav','r')
context.clf = pickle.load(file)


##############################################
def Order( tick , shares_order , index , shares_price):
    [date , equity, price, shares, order] = context.df_trade_log.iloc[-1]

    equity = tick
    shares = shares + shares_order
    price = shares_price
    if shares_order > 0:
        order = 1  #buy
    else:
        order = -1 #sell

    tradelog=[ index , equity, price, shares, order]
    context.df_trade_log = context.df_trade_log.append(pd.Series(tradelog, index=['date' , 'Positions', 'equity prices', 'shares' , 'order']),ignore_index=True)
    return order

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
    for index ,row in df_data.iterrows():
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
                close_n = df_data['Close'].iloc[index] - df_data['Close'].iloc[index - m]
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
    else:


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

###### this is what we change
def Trade_signal_generator(index):
    tick = context.tick
    end = index.date() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=69)
    df_data = data.DataReader(tick, 'google', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    df_data = Indicator_soup(df_data)
    df_data = df_data.reset_index()
    df_data_WD = Weekday(df_data)
    df_data = df_data.drop('Date', 1)
    df_data = Moar_indicators(df_data)
    df_data = Normalize(df_data)
    df_data['Weekday'] = df_data_WD
    IN = df_data.iloc[-1].tolist()
    context.inn = IN

    if context.method == 'reg':
        proba = context.clf.predict(np.asarray(IN).reshape(1, -1))[0]
        print proba
        if proba < -1:
            signal = 1
        elif proba > 1:
            signal = -1
        else:
            signal = 0

    elif context.method == 'class':
        proba = context.clf.predict_proba(np.asarray(IN).reshape(1, -1))[0][0]
        print proba
        if proba > 0.80:
            signal = -1
        elif proba < 0.10:
            signal = 1
        else:
            signal = 0
    else:
        signal = 0
        print ' error no method set'
        raise

    return signal, proba

###### this is what we change

trade_start =  [ df_data.index.values[0] , 0  , 0 , 0 , 0  ]
Portfolio_start =  [ df_data.index.values[0] , 0 , 20000 , 20000 , 0 , 0 , 0 , 0]
context.df_trade_log = pd.DataFrame( data =[trade_start] , columns=['date' , 'Positions', 'equity prices', 'shares' , 'order'])
context.df_portfolio = pd.DataFrame( data =[Portfolio_start ] , columns=['date' , 'Positions', 'Porfolio value', 'capital', 'equity prices', 'shares' , 'order' , 'proba'])

tick=  context.tick
five_day_counter = 0

######### ---------------------------------- Backtest Module -----------------------
for index , data_ in df_data.iterrows():
    [ date , equity , portfolio_val , capital_ava , price , shares , order , proba_l] = context.df_portfolio.iloc[-1]

    price = round(data_['Close'],2)
    signal , proba = Trade_signal_generator(index)


    ########## Insert signal generation or trade method here.

    ordersize = math.floor(capital_ava / price)
    if ordersize < 1 :
        ordersize = 1

    if signal > 0 :
        if capital_ava > price*ordersize:
            order = Order(tick, ordersize, index, price)
            equity = tick
            shares = ordersize + shares
            capital_ava = capital_ava - price * ordersize
        five_day_counter = 0

    elif signal < 0 :
        order = Order(tick, -shares, index, price)
        equity = 0
        capital_ava = capital_ava + price * shares
        shares = 0
        five_day_counter = 0

    else:
        five_day_counter = five_day_counter +1
        order = 0

    if five_day_counter == 5 :
        order = Order(tick, -shares, index, price)
        equity = 0
        capital_ava = capital_ava + price * shares
        shares = 0
        five_day_counter = 0

    porfolio_val = capital_ava + price * shares
    print portfolio_val

    portfolio_log = [ index , equity ,  porfolio_val , capital_ava , price , shares , order, proba]
    context.df_portfolio = context.df_portfolio.append(pd.Series(portfolio_log, index=[ 'date' , 'Positions', 'Porfolio value', 'capital', 'equity prices', 'shares','order','proba']) ,  ignore_index=True)


buyandhold = 20000 / df_data['Close'].iloc[0]
porfolio_out = context.df_portfolio['Porfolio value'].tolist()
del porfolio_out[0]

out = []
for index , data_ in context.df_portfolio.iterrows():
    proba = context.df_portfolio['proba'].iloc[index]
    price = context.df_portfolio['equity prices'].iloc[index]
    try:
        prices_next_5 = context.df_portfolio['equity prices'].iloc[index+1:index+5]
    except:
        prices_next_5 = [ price , price , price , price]

    if any(x > price for x in prices_next_5):
        out.append(1)
    else:
        out.append(-1)

context.df_portfolio['check'] = out

np.corrcoef(context.df_portfolio['check'] , context.df_portfolio['proba'])
print context.df_portfolio

print context.df_portfolio['Porfolio value'] ,(df_data['Close']*buyandhold)

'''
B = [1]
for a in B:
    print 1
    plt.plot(porfolio_out)
    plt.plot((df_data['Close']*buyandhold).tolist())
    plt.show()
'''