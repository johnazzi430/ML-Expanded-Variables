ML Expanded Variable prediction 

This is my first stab at utilizing machine learning and neural networks to develop a trading strategy. The strategy is simple. The output [Y] is a binomial it can either be 1 or -1. We want to see if we can accuratly predict if sometime in the next 5 days a stock will be worth more or less than it is now. stock price of (n+1:n+5) > stock price at (n)

To do this we are going to train a neural network on a basket of inputs. from publicly available data. Using the pandas datareader module we can download historical data from google. Why do we want to use a NN for this? Well its complicated. By all traditional understandings of the stock market we get the statistically verifible impression that stocks are random, and no set of quantifiable variables looked at to this point can accuratly predict the stock price of of tomorrow. ML would only tell us what we already know, that stocks prices are a random stocastic timeseries that describes a dynamic system of millions of individual actors making decisions on the best information they have. All economic theory supports this, and no one not even the people on wallstreet can generate positive alpha over the long run, other than their fee structure. 

so why try? well.... as we stated a stock price is based on the information that all participantzs in the market have at any given time. weather its a set of humans, or a set of robots programed by humans they can only look at so much data at once. It has been shown that opportunities for arbitrage exists when you are looking at a descriptive set of data that intangibly reltated to the stock price. for example Google was able to use a very simple algorithm to show that based on search data for certain terms you can predict if the S&P index will increase or decrease over a given week. 

Thats not my goal for this code. This code is much simplier, but we are trying to achive the same thing. Our data is readily available, it does not require a lot of effort or cost to get. so keep expectations low. However this is where we leverage the power of a dedicated neural network. 90% of trades are making decisions solely on this simple set of data, our goal is to take this log of decision making, and find the patterns that only a NN has the patience, endurance and thoroughness to sort. 

Next we create our first basket of variables: Using Talib we go through the standard set of technical indicators. RSI , BB , SMA , EMA, MACD. we also create additional variables that provide delta values between the current price, and the future price. We also want, just for fun, to see the "snake" of previous price changes, a neural network might be able to pick up an obscure pattern that does a good job at prediction. 

Normalizing. 
There are tons of methods to normalize data, so far none have worked for me. So we are going to use a normalizing method developed by stock analysts, but modified. Relative strength indicator is a normalized ( values between 0 and 100 ) based on the relation of the current price against a lookback period. The average is always moving, and we want to know momentum really. because what does it matter if a stock moves from $110 to $111 if the mean of the stock price over the past 5 years is $30. the mean timeframe needs to be shorter. 

Setting up the learning. This is a challenge. 


Getting out a prediction


translating it:

backtesting
Backtesting is done on a new file. I decided to seperate them to make the pre-processign easier to tease out. The files are long already and its too much to wait for an ultimately poor answer. 

Set the backtesters context to the same settings as 



Plugins required:

datetime
numpy
pandas
urllib2
math
re
pandas_datareader ( https://pandas-datareader.readthedocs.io/en/latest/ )
