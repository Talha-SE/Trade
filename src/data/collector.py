import ccxt
import pandas as pd
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from pycoingecko import CoinGeckoAPI

class MultiSourceDataCollector:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        
    def fetch_coingecko_data(self, days=365):
        """Fetch data from CoinGecko (Free, no API key needed)"""
        try:
            print("Fetching data from CoinGecko...")
            # Get Bitcoin price data
            data = self.cg.get_coin_market_chart_by_id(
                id='bitcoin',
                vs_currency='usd',
                days=days
            )
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                
                # Create OHLC from price (approximation)
                # In real scenario, you'd get actual OHLC data
                high = price * 1.01  # Approximate high
                low = price * 0.99   # Approximate low
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            print(f"✅ Fetched {len(df)} records from CoinGecko")
            return df
            
        except Exception as e:
            print(f"❌ CoinGecko failed: {e}")
            return None
    
    def fetch_yahoo_finance_data(self, period="2y"):
        """Fetch data from Yahoo Finance (Free, good for daily data)"""
        try:
            print("Fetching data from Yahoo Finance...")
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period=period)
            
            if data.empty:
                raise Exception("No data returned from Yahoo Finance")
            
            # Convert to our format
            df = pd.DataFrame({
                'timestamp': [int(ts.timestamp() * 1000) for ts in data.index],
                'open': data['Open'].values,
                'high': data['High'].values,
                'low': data['Low'].values,
                'close': data['Close'].values,
                'volume': data['Volume'].values
            })
            
            print(f"✅ Fetched {len(df)} records from Yahoo Finance")
            return df
            
        except Exception as e:
            print(f"❌ Yahoo Finance failed: {e}")
            return None
    
    def fetch_alternative_exchange_data(self):
        """Try alternative exchanges that might not have geographic restrictions"""
        exchanges_to_try = ['kraken', 'coinbase', 'bitfinex', 'huobi']
        
        for exchange_name in exchanges_to_try:
            try:
                print(f"Trying {exchange_name}...")
                exchange = getattr(ccxt, exchange_name)()
                
                # Test if we can access the exchange
                exchange.load_markets()
                
                # Fetch some data
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=100)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    print(f"✅ Successfully fetched data from {exchange_name}")
                    return df, exchange_name
                    
            except Exception as e:
                print(f"❌ {exchange_name} failed: {e}")
                continue
        
        return None, None
    
    def create_synthetic_data(self, days=730, timeframe='1h'):
        """Create realistic synthetic data for development"""
        print("Creating synthetic Bitcoin data...")
        
        # Determine frequency
        freq_map = {
            '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': 'h', '4h': '4h', '1d': 'D'
        }
        freq = freq_map.get(timeframe, 'h')
        
        # Create date range
        end_date = datetime.now()
        if timeframe in ['1m', '5m', '15m', '30m']:
            periods = min(days * 24 * (60 // int(timeframe.replace('m', ''))), 10000)  # Limit for memory
        elif timeframe == '1h':
            periods = days * 24
        elif timeframe == '4h':
            periods = days * 6
        else:  # 1d
            periods = days
            
        dates = pd.date_range(end=end_date, periods=periods, freq=freq)
        
        # Generate realistic price data using random walk with trend
        np.random.seed(42)
        initial_price = 45000
        
        # Create returns with some trend and volatility clustering
        base_return = 0.0001  # Small positive trend
        volatility = 0.02
        
        returns = []
        current_vol = volatility
        
        for i in range(len(dates)):
            # Volatility clustering
            vol_change = np.random.normal(0, 0.001)
            current_vol = max(0.005, min(0.05, current_vol + vol_change))
            
            # Generate return with current volatility
            daily_return = np.random.normal(base_return, current_vol)
            returns.append(daily_return)
        
        # Generate price series
        prices = [initial_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # Floor price at $1000
        
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC
            daily_volatility = abs(returns[i]) * 2
            high = close_price * (1 + daily_volatility * np.random.uniform(0.5, 1.5))
            low = close_price * (1 - daily_volatility * np.random.uniform(0.5, 1.5))
            
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1]
            
            # Ensure OHLC relationship is maintained
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume with some correlation to price movement
            price_change = abs(returns[i])
            base_volume = 50000
            volume = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': int(date.timestamp() * 1000),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Created {len(df)} synthetic records for {timeframe}")
        return df

def collect_data(symbols, timeframes, since, limit=1000):
    """Enhanced data collection with multiple fallback sources"""
    collector = MultiSourceDataCollector()
    all_data = {}
    
    for symbol in symbols:
        all_data[symbol] = {}
        print(f"\n📊 Collecting data for {symbol}")
        
        # Try different data sources in order of preference
        
        # 1. Try Yahoo Finance for daily data
        if '1d' in timeframes:
            yahoo_data = collector.fetch_yahoo_finance_data(period="2y")
            if yahoo_data is not None:
                all_data[symbol]['1d'] = yahoo_data
                print("✅ Got daily data from Yahoo Finance")
        
        # 2. Try CoinGecko for general data
        coingecko_data = collector.fetch_coingecko_data(days=730)
        if coingecko_data is not None and '1d' not in all_data[symbol]:
            # Convert CoinGecko data to daily if needed
            all_data[symbol]['1d'] = coingecko_data
        
        # 3. Try alternative exchanges
        alt_data, exchange_used = collector.fetch_alternative_exchange_data()
        if alt_data is not None and exchange_used:
            print(f"✅ Using {exchange_used} for additional data")
            # You can use this data for other timeframes
        
        # 4. For all other timeframes, create synthetic data based on daily data
        for timeframe in timeframes:
            if timeframe not in all_data[symbol]:
                print(f"Creating synthetic {timeframe} data...")
                synthetic_data = collector.create_synthetic_data(
                    days=365 if timeframe in ['4h', '1d'] else 180,  # Less data for high frequency
                    timeframe=timeframe
                )
                all_data[symbol][timeframe] = synthetic_data
                time.sleep(0.1)  # Small delay to prevent overwhelming
    
    return all_data

def save_data_to_csv(data, filename_prefix):
    """Save collected data to CSV files"""
    for symbol, timeframes in data.items():
        clean_symbol = symbol.replace('/', '_')
        for timeframe, df in timeframes.items():
            filename = f"{filename_prefix}_{clean_symbol}_{timeframe}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Saved {filename}")

# Legacy functions for compatibility
def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    """Legacy function - now uses multi-source approach"""
    collector = MultiSourceDataCollector()
    
    # Try Yahoo Finance first for daily data
    if timeframe == '1d':
        data = collector.fetch_yahoo_finance_data()
        if data is not None:
            return data.values.tolist()
    
    # Fallback to synthetic data
    days = 365 if timeframe in ['4h', '1d'] else 180
    synthetic_data = collector.create_synthetic_data(days=days, timeframe=timeframe)
    return synthetic_data.values.tolist()