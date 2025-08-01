# RL Options Trading Bot

This repository contains a Streamlit-based application powered by a Deep Q-Learning (DQN) reinforcement learning agent for trading US stock options using real-time data and paper trading via Alpaca.

## 📂 Project Structure

```
.
├── app/                     # Streamlit frontend
│   └── app.py
├── rl_model/                # Reinforcement learning training and environment
│   ├── backtest.py
│   ├── model_utils.py
│   ├── train_rl_model.py
│   └── trading_env.py
├── utils/                   # API connectors and data utilities
│   ├── alpaca_utils.py
│   ├── config.py
│   └── option_data_utils.py
├── data/                    # (Optional) Local cache or datasets
├── requirements.txt
└── README.md
```

## 🧠 Features

- RL model trained on option prices, Greeks, and macroeconomic indicators
- Streamlit frontend for signal monitoring and paper trading
- Backtesting script to visualize portfolio growth
- Alpaca and Tradier API integration

## 🚀 Getting Started

1. Clone the repo
2. Set your API keys in `utils/config.py`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the RL model:

```bash
python rl_model/train_rl_model.py
```

5. Run the app:

```bash
streamlit run app/app.py
```

## 📈 Backtest

```bash
python rl_model/backtest.py
```
