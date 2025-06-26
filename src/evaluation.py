import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_predictions(y_true, y_pred, ticker, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{ticker} - {model_name} Forecast')
    plt.legend()
    os.makedirs(f'results/plots/{ticker}', exist_ok=True)
    plt.savefig(f'results/plots/{ticker}/{model_name}_forecast.png')
    plt.close()

def evaluate_performance(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'R2': r2}

def save_evaluation_report(metrics, ticker, model_name):
    os.makedirs(f'results/logs/{ticker}', exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(f'results/logs/{ticker}/{model_name}_evaluation.csv', index=False)
