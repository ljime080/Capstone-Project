from trading_env import OptionsTradingEnv
from stable_baselines3 import DQN
from option_data_utils import get_option_data
import matplotlib.pyplot as plt

def run_backtest():
    data = get_option_data()
    env = OptionsTradingEnv(data)
    model = DQN.load("dqn_options_trader")

    obs = env.reset()
    assets = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        row = env.data.iloc[env.current_step]
        price = row["close"]
        total_value = env.cash + env.holding * price
        assets.append(total_value)
        if done:
            break

    plt.plot(assets)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Total Value")
    plt.grid()
    plt.tight_layout()
    plt.savefig("/mnt/data/backtest_result.png")
