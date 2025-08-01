import os
os.environ['STREAMLIT_WATCH_USE_POLLING'] = 'true'

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from binance.client import Client
import streamlit.components.v1 as components
import joblib

# --- Model Definition with DyT and Best Configs ---
class MultiTaskTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, init_a=0.95):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm1': nn.LayerNorm(d_model, elementwise_affine=True),
                'norm2': nn.LayerNorm(d_model, elementwise_affine=True),
                'dropout': nn.Dropout(dropout)
            })
            self.blocks.append(block)
        
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.output_layer = nn.Linear(d_model, 3)

        for block in self.blocks:
            nn.init.constant_(block['norm1'].weight, init_a)
            nn.init.constant_(block['norm2'].weight, init_a)

    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            attn_output = block['attn'](x, x, x)[0]
            x = block['norm1'](x + block['dropout'](attn_output))
            ffn_output = block['ffn'](x)
            x = block['norm2'](x + block['dropout'](ffn_output))
        
        x = self.final_norm(x)
        return self.output_layer(x[:, -1, :])

# --- Secrets ---
API_KEY = st.secrets["binance_testnet"]["API_KEY"]
API_SECRET = st.secrets["binance_testnet"]["API_SECRET"]
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# --- Feature Preprocessing ---
def preprocess(df):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)

    features = [
        'Close', 'Volume', 'MA_20', 'MA_50', 'EMA_20',
        'BB_upper', 'BB_lower', 'RSI', 'Volatility',
        'Close_lag_1', 'Close_lag_2'
    ]
    return df[features].tail(60)

# --- Load Scalers ---
try:
    feat_scaler = joblib.load("models/feature_scaler.pkl")
    tgt_scaler = joblib.load("models/target_scaler.pkl")  # Assumed fitted on ['Close', 'Volatility']
except FileNotFoundError:
    st.error("Scaler files not found. Please ensure 'models/feature_scaler.pkl' and 'models/target_scaler.pkl' exist in the app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scalers: {str(e)}")
    st.stop()

def scale_features(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[
            'Close', 'Volume', 'MA_20', 'MA_50', 'EMA_20',
            'BB_upper', 'BB_lower', 'RSI', 'Volatility',
            'Close_lag_1', 'Close_lag_2'
        ])
    X_scaled = feat_scaler.transform(X).astype('float32')
    st.write("Scaled Input Range (min, max):", X_scaled.min(), X_scaled.max())  # Debug
    return X_scaled

def rescale_prediction(y_scaled):
    # Rescale only Close and Volatility, handle Direction separately
    y_rescaled = np.zeros(3)
    y_rescaled[:2] = tgt_scaler.inverse_transform([y_scaled[:2]])[0]  # Close, Volatility
    y_rescaled[2] = torch.sigmoid(torch.tensor(y_scaled[2])).item()    # Direction probability
    return y_rescaled

# --- Binance Live Fetcher ---
def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1m', limit=120):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Vol", "Taker Buy Quote Vol", "Ignore"])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df[["Open Time", "Close", "Volume"]]

# --- Financial Metrics ---
def calculate_metrics(trades):
    if not trades:
        return 0.0, 0.0, 0
    returns = [trade['pnl'] for trade in trades]
    returns_series = pd.Series(returns)
    
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe = mean_return / (std_return + 1e-8) * np.sqrt(365*24*60) if std_return != 0 else 0.0
    total_pnl = sum(returns)
    wins = len([r for r in returns if r > 0])
    win_rate = (wins / len(returns)) * 100 if returns else 0
    
    return sharpe, total_pnl, win_rate

# --- Online Learning ---
def online_train(model, X_scaled, epochs=1, lr=7.146626141272202e-05):
    X = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    y = model(X).detach()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    model.eval()
    return f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | Training loss: {loss.item():.6f}"

# --- Load Model ---
@st.cache_resource
def load_model():
    model = MultiTaskTransformer(input_size=11, d_model=64, nhead=4, num_layers=2, dropout=0.1, init_a=0.95)
    state_dict = torch.load("models/model_optuna_best.pth", map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('blocks.', 'encoder.layers.')
        new_key = new_key.replace('attn.', 'self_attn.')
        new_key = new_key.replace('ffn.0.', 'linear1.')
        new_key = new_key.replace('ffn.2.', 'linear2.')
        new_key = new_key.replace('norm1.', 'norm1.')
        new_key = new_key.replace('norm2.', 'norm2.')
        new_key = new_key.replace('final_norm.', 'norm.')
        new_key = new_key.replace('head.', 'output_layer.')
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# --- Notification System ---
def send_notification(message, signal_type):
    signal_colors = {
        'buy': '#2ECC71',
        'sell': '#E74C3C',
        'stop_loss': '#FF4136',
        'take_profit': '#2ECC71'
    }
    color = signal_colors.get(signal_type, '#3498DB')
    js = f"""
    <script>
        if ('Notification' in window && Notification.permission === 'granted') {{
            new Notification('{message}', {{
                body: 'BTC/USDT Trading Signal',
                icon: 'https://cryptologos.cc/logos/bitcoin-btc-logo.png'
            }});
        }}
    </script>
    """
    components.html(js, height=0)
    st.markdown(f"""
    <div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        {message}
    </div>
    """, unsafe_allow_html=True)

# --- Streamlit UI ---
st.set_page_config("Minutely BTC Trading Dashboard", layout="wide")

if 'predicted_history' not in st.session_state:
    st.session_state['predicted_history'] = []
if 'training_log' not in st.session_state:
    st.session_state['training_log'] = []
if 'trades' not in st.session_state:
    st.session_state['trades'] = []
if 'position' not in st.session_state:
    st.session_state['position'] = None
if 'balance' not in st.session_state:
    st.session_state['balance'] = 10000.0

st.title("Live Bitcoin Minutely Trading Dashboard")

st.sidebar.header("Trading Parameters")
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 5.0, 1.0) / 100
take_profit_pct = st.sidebar.slider("Take Profit (%)", 0.1, 5.0, 2.0) / 100
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0) / 100
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0.0, 100.0, 60.0) / 100

model = load_model()
raw_df = fetch_binance_ohlcv()
X_raw = preprocess(raw_df)
X_scaled = scale_features(X_raw.values)

if X_raw.shape[0] >= 60:
    log_line = online_train(model, X_scaled)
    st.session_state['training_log'].append(log_line)
    if len(st.session_state['training_log']) % 10 == 0:
        torch.save(model.state_dict(), "models/model_minutely_full_new.pth")

st.subheader("Recent BTC/USDT Minutely Data")
st.dataframe(X_raw.tail(), use_container_width=True)

st.markdown("### Model Prediction")
st.code(f"Fetched rows: {len(raw_df)}")
st.code(f"Preprocessed rows: {X_raw.shape[0]}")

if X_raw.shape[0] < 60:
    st.warning("Not enough data to make a prediction.")
else:
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction_scaled = model(input_tensor).squeeze().numpy()
        prediction = rescale_prediction(prediction_scaled)
        close = prediction[0] - 75673 #temporary cap
        vol = abs(prediction[1])
        direction_prob = prediction[2]
        direction = 1 if direction_prob > 0.5 else 0
        confidence = max(direction_prob, 1 - direction_prob)
        current_price = raw_df["Close"].iloc[-1]
        next_time = raw_df["Open Time"].iloc[-1] + pd.Timedelta(minutes=1)
        st.session_state['predicted_history'].append((next_time, close))
        st.write("Predicted (Scaled):", prediction_scaled)
        st.write("Predicted (Rescaled):", prediction)

        account_balance = st.session_state['balance']
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / (stop_loss_pct * current_price)
        position_size = min(position_size, account_balance / current_price)

        if direction == 1 and st.session_state['position'] is None and confidence >= confidence_threshold:
            st.session_state['position'] = {
                'type': 'long',
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': current_price * (1 - stop_loss_pct),
                'take_profit': current_price * (1 + take_profit_pct)
            }
            send_notification(f"Buy Signal: Entered long at ${current_price:,.2f}, Size: {position_size:.4f} BTC, Confidence: {confidence:.2%}", 'buy')
        
        elif direction == 0 and st.session_state['position'] is None and confidence >= confidence_threshold:
            st.session_state['position'] = {
                'type': 'short',
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': current_price * (1 + stop_loss_pct),
                'take_profit': current_price * (1 - take_profit_pct)
            }
            send_notification(f"Sell Signal: Entered short at ${current_price:,.2f}, Size: {position_size:.4f} BTC, Confidence: {confidence:.2%}", 'sell')

        if st.session_state['position']:
            pos = st.session_state['position']
            if pos['type'] == 'long':
                if current_price <= pos['stop_loss']:
                    pnl = (current_price - pos['entry_price']) * pos['size']
                    st.session_state['trades'].append({'pnl': pnl, 'type': 'long', 'exit_reason': 'stop_loss'})
                    st.session_state['balance'] += pnl
                    send_notification(f"Stop Loss Hit: Exited long at ${current_price:,.2f}, P/L: ${pnl:,.2f}", 'stop_loss')
                    st.session_state['position'] = None
                elif current_price >= pos['take_profit']:
                    pnl = (current_price - pos['entry_price']) * pos['size']
                    st.session_state['trades'].append({'pnl': pnl, 'type': 'long', 'exit_reason': 'take_profit'})
                    st.session_state['balance'] += pnl
                    send_notification(f"Take Profit Hit: Exited long at ${current_price:,.2f}, P/L: ${pnl:,.2f}", 'take_profit')
                    st.session_state['position'] = None
            else:
                if current_price >= pos['stop_loss']:
                    pnl = (pos['entry_price'] - current_price) * pos['size']
                    st.session_state['trades'].append({'pnl': pnl, 'type': 'short', 'exit_reason': 'stop_loss'})
                    st.session_state['balance'] += pnl
                    send_notification(f"Stop Loss Hit: Exited short at ${current_price:,.2f}, P/L: ${pnl:,.2f}", 'stop_loss')
                    st.session_state['position'] = None
                elif current_price <= pos['take_profit']:
                    pnl = (pos['entry_price'] - current_price) * pos['size']
                    st.session_state['trades'].append({'pnl': pnl, 'type': 'short', 'exit_reason': 'take_profit'})
                    st.session_state['balance'] += pnl
                    send_notification(f"Take Profit Hit: Exited short at ${current_price:,.2f}, P/L: ${pnl:,.2f}", 'take_profit')
                    st.session_state['position'] = None

sharpe, total_pnl, win_rate = calculate_metrics(st.session_state['trades'])
st.markdown("### Trading Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("Total P/L (USDT)", f"${total_pnl:,.2f}")
col3.metric("Win Rate (%)", f"{win_rate:.1f}%")
col4.metric("Account Balance (USDT)", f"${st.session_state['balance']:,.2f}")

st.markdown("""
<style>
.metric-container {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    margin-top: 4px;
}
.up { color: #2ECC71; }
.down { color: #E74C3C; }
</style>
""", unsafe_allow_html=True)

direction_icon = "⬆️" if direction else "⬇️"
direction_class = "up" if direction else "down"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Close Price
        <div class="metric-value">${close:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Volatility
        <div class="metric-value">{vol:.6f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-container">
        Predicted Direction
        <div class="metric-value {direction_class}">{direction_icon} {'Up' if direction else 'Down'} (Conf: {confidence:.2%})</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Actual vs Predicted Close Price (Auto-refreshes every 60s)")
timestamps = raw_df["Open Time"].tail(60).tolist()
actual_close = raw_df["Close"].tail(60).tolist()
next_time = timestamps[-1] + pd.Timedelta(minutes=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=timestamps, y=actual_close, mode="lines+markers", name="Actual Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=[next_time], y=[close], mode="markers+text", marker=dict(color="red", size=10), name="Next Prediction"))

if len(st.session_state['predicted_history']) > 1:
    pred_times, pred_closes = zip(*st.session_state['predicted_history'])
    fig.add_trace(go.Scatter(x=list(pred_times), y=list(pred_closes), mode="lines+markers", line=dict(color="red", width=2, dash="dash"), name="Predicted Line"))

fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), showlegend=True, xaxis_title="Time", yaxis_title="BTC/USDT Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Trade History")
if st.session_state['trades']:
    trade_df = pd.DataFrame(st.session_state['trades'])
    trade_df['time'] = [pd.Timestamp.now() for _ in range(len(trade_df))]
    st.dataframe(trade_df[['time', 'type', 'pnl', 'exit_reason']], use_container_width=True)
else:
    st.info("No trades executed yet.")

st.markdown("### 🔧 Online Training Log")
with st.expander("Show training updates", expanded=False):
    if st.session_state['training_log']:
        for log in reversed(st.session_state['training_log'][-10:]):
            st.markdown(f"- {log}")
    else:
        st.info("No training events yet.")

st_autorefresh(interval=60000, key="refresh")

components.html("""
<script>
if ('Notification' in window && Notification.permission !== 'granted') {
    Notification.requestPermission();
}
</script>
""", height=0)