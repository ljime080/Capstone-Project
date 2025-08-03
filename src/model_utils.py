import os
from trading_env import OptionsTradingEnv
from stable_baselines3 import DQN
from option_data_utils import get_option_data

MODEL_PATH = "dqn_options_trader.zip"

def generate_trade_signal(data=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train it first.")

    model = DQN.load(MODEL_PATH)

    if data is None:
        data = get_option_data().tail(100)

    env = OptionsTradingEnv(data)
    obs = env.reset()

    steps = min(len(data) - 1, 99)
    for _ in range(steps):
        obs, _, done, _ = env.step(0)
        if done:
            break

    action = int(model.predict(obs, deterministic=True)[0])
    return {0: "buy", 1: "sell"}[action]
