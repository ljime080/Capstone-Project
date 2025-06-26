import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return self._next_observation()

    def _next_observation(self):
        return np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.shares_held
        ])

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        if action == 1:
            self.shares_held += self.balance // current_price
            self.balance -= self.shares_held * current_price
        elif action == 2 and self.shares_held > 0:
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.balance + self.shares_held * current_price - 10000
        return self._next_observation(), reward, done, {}
