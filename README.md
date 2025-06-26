# AI Stock Options Trading Project

This project implements a full AI system for trading the top 10 US stocks using:
- Deep Learning (LSTM & Transformer) for price forecasting
- Reinforcement Learning (PPO) for trading strategy
- Evaluation and visualization tools
- Optional hyperparameter tuning using Optuna

---

## 📦 How to Run the System

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Entire Pipeline
This downloads data, trains LSTM and RL models for all stocks:
```bash
python main.py
```

### 3. Run Optuna Optimization for LSTM
```bash
python src/optimize_lstm.py
```

---

## 📊 Mathematical Formulations

### 🔮 Deep Learning Forecasting

We model the time series forecasting problem as:

**Objective**:  
Given past `T` closing prices x_{t-T}, ..., x_{t-1}, predict the next closing price x_t.

#### 🧠 LSTM Model

Forget gate:  
![Forget Gate](docs/lstm_forget_gate.png)

Input gate:  
![Input Gate](docs/lstm_input_gate.png)

Cell update:  
![Cell Update](docs/lstm_cell_update.png)

Cell state:  
![Cell State](docs/lstm_cell_state.png)

Output gate:  
![Output Gate](docs/lstm_output_gate.png)

Hidden state:  
![Hidden State](docs/lstm_hidden_state.png)

Final output:  
![Final Output](docs/lstm_output.png)

#### 🧠 Transformer Model

Self-attention mechanism:  
![Attention](docs/attention.png)

---

### 🎮 Reinforcement Learning (RL)

#### PPO Algorithm

We define a Markov Decision Process (MDP) with:
- State s_t: stock price, indicators, and model prediction
- Action a_t: Buy, Sell, or Hold
- Reward r_t: change in portfolio value
- Policy π_θ(a_t | s_t)

Clipped policy objective:  
![PPO Loss](docs/ppo_loss.png)

Probability ratio:  
![PPO Ratio](docs/ppo_ratio.png)

---

## 📁 Directory Overview

- `data/`: Raw and processed stock data
- `models/`: Saved LSTM/Transformer models and RL agents
- `notebooks/`: Jupyter notebooks for testing/EDA
- `src/`: All core scripts (training, RL, evaluation)
- `results/`: Plots and evaluation logs
