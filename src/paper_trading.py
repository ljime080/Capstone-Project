import alpaca_trade_api as tradeapi
import numpy as np
import time
from tensorflow.keras.models import load_model
import yfinance as yf
from utils.logger import log

# Load your DL model
model = load_model("models/model.h5")

# Alpaca paper trading credentials
API_KEY = "YOUR_PAPER_API_KEY"
SECRET_KEY = "YOUR_PAPER_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

# Setup Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Trading config
SYMBOL = "AAPL"
LOOKBACK = 60  # number of past days for input
INTERVAL = "1d"
CASH_ALLOCATION = 0.1  # % of portfolio to trade per position

def get_latest_data(symbol):
    df = yf.download(symbol, period=f"{LOOKBACK+1}d", interval=INTERVAL)
    prices = df["Close"].values[-LOOKBACK:]
    return np.array(prices).reshape(1, LOOKBACK, 1)

def predict_price():
    prices = get_latest_data(SYMBOL)
    scaled = (prices - prices.min()) / (prices.max() - prices.min())
    prediction = model.predict(scaled)
    return prediction[0][0]

def get_position():
    try:
        return api.get_position(SYMBOL)
    except:
        return None

def execute_trade(predicted_price, last_price):
    direction = "buy" if predicted_price > last_price else "sell"
    cash = float(api.get_account().cash)
    qty = int((cash * CASH_ALLOCATION) / last_price)

    if direction == "buy":
        log(f"Buying {qty} shares of {SYMBOL}")
        api.submit_order(symbol=SYMBOL, qty=qty, side='buy', type='market', time_in_force='gtc')
    elif direction == "sell":
        position = get_position()
        if position:
            qty = int(float(position.qty))
            log(f"Selling {qty} shares of {SYMBOL}")
            api.submit_order(symbol=SYMBOL, qty=qty, side='sell', type='market', time_in_force='gtc')

if __name__ == "__main__":
    predicted = predict_price()
    latest = get_latest_data(SYMBOL)[0, -1, 0]
    log(f"Predicted: {predicted:.2f}, Latest: {latest:.2f}")
    execute_trade(predicted, latest)
