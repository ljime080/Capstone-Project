from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import TradingEnv
import os

def train_agent(ticker, df):
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    os.makedirs(f"models/rl/{ticker}", exist_ok=True)
    model.save(f"models/rl/{ticker}/ppo_agent")
    return model
