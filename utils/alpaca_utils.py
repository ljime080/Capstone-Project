import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

def get_alpaca_api():
    return tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        ALPACA_BASE_URL,
        api_version="v2"
    )

def get_price_data(symbol="SPY", timeframe="1D", limit=100):
    api = get_alpaca_api()
    barset = api.get_bars(symbol, timeframe, limit=limit).df
    return barset[barset['symbol'] == symbol]

def submit_order(symbol, qty, side, type="market", time_in_force="gtc"):
    api = get_alpaca_api()
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=type,
        time_in_force=time_in_force
    )
