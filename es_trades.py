import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import sys
from multiprocessing import Pool
import talib

scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv("ES_Trades.csv", parse_dates=[['Date','Time']], nrows = 10000)

df = df.set_index(['Date_Time'])

df.index = pd.to_datetime(df.index, unit='ns')
# print(df)

ticks =  df.iloc[:,0:3]
# print(ticks)

bar_intervals = ['1min','2min','3min', '4min', '5min',
                '6min','8min','10min', '9min', '12min',
                '15min','16min','20min', '25min']
 
# bar_intervals = ['1min']

def create_bars(interval):
    bars = ticks['Price'].resample('1min').ohlc()

    bars['amplitude'] = np.absolute((bars['high']-bars['low'])*np.where(bars['close']>bars['open'], 1, -1))
    bars['bar_body'] = bars['close'] - bars['open']
    print(bars.iloc[:5])
    print(np.mean(bars, axis=0))

    bars = clean_dataset(bars)

    try:
        bars['trans_close'] = scaler.fit_transform(bars[['close']])
        bars['trans_amplitude'] = scaler.fit_transform(bars[['amplitude']])
        bars['trans_body'] = scaler.fit_transform(bars[['bar_body']])
    except:
        print("Unexpected error.", sys.exc_info()[0])
        raise

    bars['MACD'],bars['MACDsignal'],bars['MACDhist'] = talib.MACD(np.array(bars['close']),
                            fastperiod=6, slowperiod=12, signalperiod=9)   

    bars["RSI"] = talib.RSI(bars['close'], timeperiod=14)

    print(interval)
    print(bars)
    return bars

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep =  ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

bar_maps={}

for bar_interval in bar_intervals:
    bar_maps[bar_interval] = create_bars(bar_interval)

# multiprocessing.Pool().imap(create_bars, bar_intervals)
print(len(bar_maps))
