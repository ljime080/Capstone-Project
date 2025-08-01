from stable_baselines3 import DQN
from option_data_utils import get_option_data
from trading_env import OptionsTradingEnv

data = get_option_data()
env = OptionsTradingEnv(data)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("dqn_options_trader")
