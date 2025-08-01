import requests
import pandas as pd
from config import TRADIER_API_KEY, TRADIER_BASE_URL
from datetime import datetime, timedelta
import yfinance as yf

def get_macro_indicators(start_date, end_date):
    vix = yf.download("^VIX", start=start_date, end=end_date)['Close'].rename("VIX")
    cpi = yf.download("CPIAUCSL", start=start_date, end=end_date)['Close'].rename("CPI")
    fed_rate = yf.download("FEDFUNDS", start=start_date, end=end_date)['Close'].rename("FedFunds")
    macro = pd.concat([vix, cpi, fed_rate], axis=1).ffill()
    return macro

def get_option_chain(symbol="SPY", expiration=None, strike=None, option_type="call"):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": "true",
        "strike": strike,
        "option_type": option_type
    }
    headers = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if "options" not in data or data["options"] is None:
        return pd.DataFrame()

    rows = data["options"]["option"]
    return pd.DataFrame(rows)

def get_option_data(symbol="SPY"):
    end = datetime.today()
    start = end - timedelta(days=30)
    expiration = (end + timedelta(days=10)).strftime("%Y-%m-%d")

    spot_price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
    atm_strike = round(spot_price / 5) * 5

    df = get_option_chain(symbol, expiration, strike=atm_strike)

    if df.empty:
        raise ValueError("Failed to fetch option chain")

    df['date'] = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    df = df.rename(columns={
        "last": "close",
        "implied_volatility": "iv"
    })

    keep_cols = ["date", "close", "iv", "delta", "gamma", "theta", "vega"]
    df = df[keep_cols]

    macro = get_macro_indicators(start, end)
    df = df.merge(macro, left_on="date", right_index=True, how="left").ffill()
    return df
