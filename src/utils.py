import joblib
import yfinance as yf
import pandas as pd 
import numpy as np
from datetime import datetime as dt



def download_data(ticker, start_date = None, end_date=None , interval='1d'):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    interval (str): Data interval ('1d', '1h', etc.). Default is '1d'.
    
    Returns:
    pd.DataFrame: DataFrame containing the stock data.
    """
    if start_date is None:
        start_date = "2010-01-01"

    if end_date is None:
        end_date = dt.now().strftime('%Y-%m-%d')


    if ticker == None:
        raise ValueError("Ticker symbol cannot be None")
        return pd.DataFrame()
    else:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        return data


def save_scaler(scaler, path):

    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
