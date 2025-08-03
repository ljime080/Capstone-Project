import gym
import numpy as np
from gym import spaces

class OptionsTradingEnv(gym.Env):
    def __init__(self, data, initial_cash=10000, max_position=1):
        super(OptionsTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_position = max_position

        # Two actions: 0 = buy, 1 = sell
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.holding = 0
        self.buy_price = 0
        self.total_asset = self.initial_cash
        return self._get_obs()

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row["close"],
            row["iv"],
            row["delta"],
            row["gamma"],
            row["theta"],
            row["vega"],
            row["VIX"],
            row["CPI"],
            row["FedFunds"],
            self.holding
        ], dtype=np.float32)

        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row["close"]
        reward = 0

        if action == 0 and self.holding < self.max_position and self.cash >= price:
            self.holding += 1
            self.cash -= price
            self.buy_price = price  # Track entry price
        elif action == 1 and self.holding > 0:
            profit = price - self.buy_price
            reward = profit / self.buy_price  # Reward profit, penalize loss
            self.cash += price
            self.holding -= 1
        else:
            reward = -0.001  # small penalty for no-op or invalid action

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        self.current_step = min(self.current_step, len(self.data) - 1)

        current_price = self.data.iloc[self.current_step]["close"]
        self.total_asset = self.cash + self.holding * current_price
        reward = np.clip(reward, -1, 1)

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Cash: {self.cash:.2f} | Holding: {self.holding} | Total Asset: {self.total_asset:.2f}")
