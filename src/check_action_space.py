import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from trading_env import OptionsTradingEnv
from option_data_utils import get_option_data
import pandas as pd
from collections import Counter

# Load data
data = get_option_data("SPY")
data.fillna(0, inplace=True)
for col in ["close", "iv", "delta", "gamma", "theta", "vega", "VIX", "CPI", "FedFunds"]:
    if col not in data.columns:
        data[col] = 0.0

# Create env and load model
env = OptionsTradingEnv(data)
model = DQN.load("dqn_options_trader")

# Track actions
actions = []
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    actions.append(int(action))
    obs, _, done, _ = env.step(action)
    if done:
        break

# Plot
counts = Counter(actions)
plt.bar(["Buy", "Sell"], [counts.get(0, 0), counts.get(1, 0)])
plt.title("Action Distribution After Training")
plt.ylabel("Action Count")
plt.grid(True)
plt.show()