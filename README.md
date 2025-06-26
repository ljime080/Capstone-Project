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
Given past `T` closing prices \( x_{t-T}, ..., x_{t-1} \), predict the next closing price \( x_t \).

#### 🧠 LSTM Model
An LSTM maintains a hidden state \( h_t \) and cell state \( c_t \) updated as:
- Forget gate: \( f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \)
- Input gate: \( i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \)
- Cell update: \( \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \)
- Cell state: \( c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \)
- Output gate: \( o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \)
- Hidden state: \( h_t = o_t \odot \tanh(c_t) \)

Final output:  
\[
\hat{x}_t = W_{out} h_t + b_{out}
\]

#### 🧠 Transformer Model
- Uses self-attention to capture dependencies:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where \( Q = XW^Q, K = XW^K, V = XW^V \)

Final output is passed through a feed-forward network to predict \( \hat{x}_t \).

---

### 🎮 Reinforcement Learning (RL)

#### PPO Algorithm

We define a Markov Decision Process (MDP) with:
- State \( s_t \): stock price, indicators, and model prediction
- Action \( a_t \): Buy, Sell, or Hold
- Reward \( r_t \): change in portfolio value
- Policy \( \pi_\theta(a_t | s_t) \)

**Objective**:  
Maximize expected return \( \mathbb{E}[\sum_t \gamma^t r_t] \)

PPO optimizes the surrogate loss:
\[
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
\]
Where:
- \( r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \)
- \( \hat{A}_t \): Advantage estimate
- \( \epsilon \): Clipping parameter

---

## 📁 Directory Overview

- `data/`: Raw and processed stock data
- `models/`: Saved LSTM/Transformer models and RL agents
- `notebooks/`: Jupyter notebooks for testing/EDA
- `src/`: All core scripts (training, RL, evaluation)
- `results/`: Plots and evaluation logs
