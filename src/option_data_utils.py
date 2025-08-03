import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta

def get_macro_indicators(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    # Get VIX from Yahoo
    vix = yf.download("^VIX", start=start_date, end=end_date)[["Close"]]
    vix.columns = ["VIX"]
    vix.index = vix.index.date  # make index just date
    vix = vix.reset_index().rename(columns={"index": "date"})

    # Try to get FRED data
    try:
        cpi = pdr.DataReader("CPIAUCSL", "fred", start_date, end_date).rename(columns={"CPIAUCSL": "CPI"})
        fed = pdr.DataReader("FEDFUNDS", "fred", start_date, end_date).rename(columns={"FEDFUNDS": "FedFunds"})

        cpi.index = cpi.index.date
        fed.index = fed.index.date

        cpi = cpi.reset_index().rename(columns={"index": "date"})
        fed = fed.reset_index().rename(columns={"index": "date"})

        # Merge everything on 'date'
        macro = vix.merge(cpi, on="date", how="left").merge(fed, on="date", how="left")
    except Exception as e:
        print(f"⚠️ Failed to fetch CPI or FedFunds: {e}")
        macro = vix.copy()
        macro["CPI"] = None
        macro["FedFunds"] = None

    # Fill missing values forward
    macro = macro.sort_values("date").ffill()
    return macro



def get_option_data(symbol="SPY" , start = None, end = None):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        raise ValueError("No options available for this symbol")


    expiration = expirations[0]  # Use nearest expiration
    option_chain = ticker.option_chain(expiration)
    calls = option_chain.calls

    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    calls = calls.loc[abs(calls['strike'] - current_price).nsmallest(5).index]

    calls["date"] = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    calls = calls.rename(columns={"lastPrice": "close"})

    keep_cols = ["date", "close", "strike", "impliedVolatility", "volume", "openInterest"]
    calls = calls[keep_cols].copy()
    calls = calls.rename(columns={"impliedVolatility": "iv"})

    calls["delta"] = 0.5
    calls["gamma"] = 0.01
    calls["theta"] = -0.02
    calls["vega"] = 0.03

    
    macro = get_macro_indicators(start, end)
    
    calls["date"] = calls["date"].dt.date  # ensure same format as macro["date"]
    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    calls = calls.merge(macro, on="date", how="left")
    
    return calls



if __name__ == "__main__":
    data = get_option_data()
    print(data.head())
    print("Option data fetched successfully.")  
