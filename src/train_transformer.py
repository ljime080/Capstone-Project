import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, seq_len):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear', shape=(self.seq_len,), initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear', shape=(self.seq_len,), initializer='uniform', trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic', shape=(self.seq_len,), initializer='uniform', trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic', shape=(self.seq_len,), initializer='uniform', trainable=True)

    def call(self, x):
        time_linear = self.weights_linear * x + self.bias_linear
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        return tf.concat([time_linear[..., tf.newaxis], time_periodic[..., tf.newaxis]], axis=-1)

def create_transformer_model(seq_len, d_model=64, num_heads=2):
    inputs = tf.keras.Input(shape=(seq_len, 1))
    time_embedding = Time2Vector(seq_len)(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)([inputs, time_embedding])
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = tf.keras.layers.Add()([x, attention_output])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_transformer(ticker_df, ticker, window_size=30):
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

    model = create_transformer_model(window_size)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    os.makedirs(f'models/lstm/{ticker}', exist_ok=True)
    model.save(f'models/lstm/{ticker}/transformer_model.h5')
    joblib.dump(scaler, f'models/lstm/{ticker}/scaler_transformer.pkl')

    return model, scaler


from src.evaluation import plot_predictions, evaluate_performance, save_evaluation_report

    # Evaluate and visualize
    y_pred = model.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)
    save_evaluation_report(metrics, ticker, "transformer")
    plot_predictions(y_test, y_pred, ticker, "transformer")
