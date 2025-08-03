from stable_baselines3 import DQN
from trading_env import OptionsTradingEnv
from option_data_utils import get_option_data
import pandas as pd

# Get fresh training data using the existing utility
data = get_option_data("SPY")  

# Fill any missing or invalid values
data.fillna(0, inplace=True)
for col in ["close", "iv", "delta", "gamma", "theta", "vega", "VIX", "CPI", "FedFunds"]:
    if col not in data.columns:
        data[col] = 0.0

# Save it for reuse or inspection
data.to_csv("option_data_train.csv", index=False)

# Initialize environment
env = OptionsTradingEnv(data)

# Train DQN agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    verbose=1,
    tensorboard_log="./logs/",
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    train_freq=1,
    gradient_steps=1,
)

model.learn(total_timesteps=10000)

# Save the trained model
model.save("dqn_options_trader")
