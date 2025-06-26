import optuna
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def objective(trial, scaled_data, ticker):
    window_size = trial.suggest_int('window_size', 10, 60)
    n_units = trial.suggest_int('n_units', 32, 128)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 30)

    X, y = create_dataset(scaled_data, window_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        LSTM(n_units, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)

    loss = model.evaluate(X_val, y_val, verbose=0)
    return loss

def run_optimization(ticker_df, ticker):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ticker_df[['Close']].values)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, scaled, ticker), n_trials=20)

    best_params = study.best_params
    window_size = best_params['window_size']
    n_units = best_params['n_units']
    learning_rate = best_params['lr']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    X, y = create_dataset(scaled, window_size)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        LSTM(n_units, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    os.makedirs(f'models/lstm/{ticker}', exist_ok=True)
    model.save(f'models/lstm/{ticker}/lstm_optuna.h5')
    joblib.dump(scaler, f'models/lstm/{ticker}/scaler_optuna.pkl')

    from src.evaluation import plot_predictions, evaluate_performance, save_evaluation_report
    y_pred = model.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)
    save_evaluation_report(metrics, ticker, "lstm_optuna")
    plot_predictions(y_test, y_pred, ticker, "lstm_optuna")

    return model, best_params
