from src.train_lstm import train_lstm
from src.train_rl_agent import train_agent
import pandas as pd
import os
from src.utils import download_data


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JPM']



def main():
    for ticker in tickers:
        print(f"Processing {ticker}")
        df = download_data(ticker)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/{ticker}.csv")
        df.head()
        # Train LSTM Model
        train_lstm(df, ticker)

        # Train RL Agent
        train_agent(ticker, df)

if __name__ == "__main__":
    main()
