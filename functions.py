import math
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import sys
from multiprocessing import Pool
import talib


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("/home/roger/PycharmProjects/es-trades/data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec


# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep =  ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def create_trans_bars(stockName):
    df = pd.read_csv("/home/roger/PycharmProjects/es-trades/data/"+stockName+".csv")
    bars: DataFrame = df.set_index(['Date'])
    print("bars : ".format(bars))

    bars['amplitude'] = np.absolute((bars['High']-bars['Low'])*np.where(bars['Close'] > bars['Open'], 1, -1))
    bars['bar_body'] = bars['Close'] - bars['Open']
    print(bars.iloc[:5])
    print(np.mean(bars, axis=0))

    bars = clean_dataset(bars)

    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    try:
        # bars['trans_close'] = scaler.fit_transform(bars[['close']])
        bars['trans_close'] = bars['Close'].diff().apply(sigmoid)
        bars['trans_amplitude'] = scaler.fit_transform(bars[['amplitude']])
        bars['trans_body'] = scaler.fit_transform(bars[['bar_body']])
    except:
        print("Unexpected error.", sys.exc_info()[0])
        raise

    bars['MACD'],bars['MACDsignal'],bars['MACDhist'] = talib.MACD(np.array(bars['Close']), fastperiod=6, slowperiod=12,
                                                                  signalperiod=9)

    bars['trans_macd'] = scaler.fit_transform(bars[['MACD']])
    bars['trans_macd_signal'] = scaler.fit_transform(bars[['MACDsignal']])
    bars['trans_macd_hist'] = scaler.fit_transform(bars[['MACDhist']])

    bars["RSI"] = talib.RSI(bars['Close'], timeperiod=14)
    bars['trans_rsi'] = scaler.fit_transform(bars[['RSI']])

    bars["ADX"] = talib.ADX(bars['High'], bars['Low'], bars['Close'])

    bars['trans_adx'] = scaler.fit_transform(bars[['ADX']])

    trans = bars[['Close', 'trans_close', 'trans_amplitude', 'trans_macd', 'trans_macd_signal', 'trans_macd_hist',
                  'trans_body', 'trans_rsi', 'trans_adx']]
    # print(type(bars))
    print("trans : {}".format(trans))
    print(trans.index)
    # return bars
    return trans
