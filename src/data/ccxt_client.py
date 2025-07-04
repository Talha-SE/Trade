from ccxt import ccxt

class CCXTClient:
    def __init__(self, exchange_name, api_key=None, secret=None):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': secret,
        })

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

    def fetch_markets(self):
        return self.exchange.load_markets()

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self):
        return self.exchange.fetch_balance()