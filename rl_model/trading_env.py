import gym
import numpy as np
from gym import spaces

class OptionsTradingEnv(gym.Env):
    def __init__(self, data, initial_cash=10000):
        super(OptionsTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.cash = self.initial_cash
        self.holding = 0
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['close'], row['iv'], row['delta'], row['gamma'],
            row['theta'], row['vega'], row['VIX'], row['CPI'], row['FedFunds']
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row['close']
        reward = 0

        if action == 1 and self.cash >= price:
            self.holding += 1
            self.cash -= price
        elif action == 2 and self.holding > 0:
            self.holding -= 1
            self.cash += price
            reward = 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Cash: {self.cash:.2f} | Holding: {self.holding}")
