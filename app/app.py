import streamlit as st
from model_utils import generate_trade_signal
from alpaca_utils import submit_order
from option_data_utils import get_option_data

st.set_page_config(page_title="Options RL Trader", layout="wide")

st.title("🤖 RL-Powered Options Trader (Paper Trading)")

symbol = st.sidebar.text_input("Symbol", value="SPY")
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)

# Load option data and show table
data = get_option_data(symbol)
st.subheader("Latest Option Data")
st.dataframe(data.tail(5))

# Generate trade signal
signal = generate_trade_signal(data)
st.success(f"Trade Signal: **{signal.upper()}**")

# Execute paper trade
if signal != "hold" and st.button(f"Execute {signal.upper()} Order"):
    submit_order(symbol=symbol, qty=qty, side=signal)
    st.info(f"{signal.upper()} order submitted for {qty} contracts of {symbol}")
