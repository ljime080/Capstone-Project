import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(ticker_df, ticker, window_size=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ticker_df[['Close']].values)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    os.makedirs(f'models/lstm/{ticker}', exist_ok=True)
    model.save(f'models/lstm/{ticker}/model.h5')
    joblib.dump(scaler, f'models/lstm/{ticker}/scaler.pkl')

    return model, scaler


from src.evaluation import plot_predictions, evaluate_performance, save_evaluation_report

    # Evaluate and visualize
    y_pred = model.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)
    save_evaluation_report(metrics, ticker, "lstm")
    plot_predictions(y_test, y_pred, ticker, "lstm")
