from trading_env import OptionsTradingEnv
from stable_baselines3 import DQN
from option_data_utils import get_option_data

model = DQN.load("dqn_options_trader")

def generate_trade_signal(data=None):
    if data is None:
        data = get_option_data().tail(100)

    env = OptionsTradingEnv(data)
    obs = env.reset()
    for _ in range(99):
        obs, _, _, _ = env.step(0)

    action = model.predict(obs, deterministic=True)[0]
    return {0: "hold", 1: "buy", 2: "sell"}[action]
