from ccxt import binance
import pandas as pd
import time

def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    exchange = binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    return ohlcv

def collect_data(symbols, timeframes, since, limit=1000):
    all_data = {}
    for symbol in symbols:
        all_data[symbol] = {}
        for timeframe in timeframes:
            print(f"Collecting data for {symbol} at {timeframe} timeframe...")
            data = fetch_ohlcv(symbol, timeframe, since, limit)
            all_data[symbol][timeframe] = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            time.sleep(1)  # To avoid hitting the rate limit
    return all_data

def save_data_to_csv(data, filename):
    for symbol, timeframes in data.items():
        for timeframe, df in timeframes.items():
            df.to_csv(f"{filename}_{symbol}_{timeframe}.csv", index=False)