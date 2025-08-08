import os
# Set environment variable at the very top, before any Streamlit commands.
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
from binance.exceptions import BinanceAPIException  # Import for specific error handling
import streamlit.components.v1 as components
import joblib

# --- Top-20 + BTC pair list ---
TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
    "SHIBUSDT", "LTCUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT",
    "BCHUSDT", "XLMUSDT", "VETUSDT", "FILUSDT", "TRXUSDT"
]

# --- Streamlit UI Configuration (MUST be the first Streamlit command) ---
st.set_page_config("Minutely Crypto Trading Dashboard", layout="wide")

# --- Model Definition with DyT and Best Configs ---
class DyT(nn.Module):
    def __init__(self, dim: int, init_a: float = 0.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (B, T, C) or (B, C)
        return self.g * torch.tanh(self.a * x) + self.b

class TransformerBlock(nn.Module):
    """Single block with DyT instead of LayerNorm"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = DyT(d_model)
        self.norm2 = DyT(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.final_norm = DyT(d_model)
        self.head = nn.Linear(d_model, 3)   # [close, vol, dir]

    def forward(self, x):
        x = self.input_proj(x)              # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x[:, -1])      # last time step
        return self.head(x)                 # (B, 3)

# --- Secrets ---
API_KEY = st.secrets["binance_testnet"]["API_KEY"]
API_SECRET = st.secrets["binance_testnet"]["API_SECRET"]
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# --- Function to fetch and update balance from Binance Testnet ---
def fetch_binance_balance():
    st.session_state['all_balances'] = {}  # Initialize dictionary to hold all relevant balances
    try:
        account_info = client.get_account()
        for asset in account_info['balances']:
            asset_name = asset['asset']
            free_balance = float(asset['free'])
            if free_balance > 0.00000001:  # Only store non-negligible balances
                st.session_state['all_balances'][asset_name] = free_balance
        
        # Update main USDT balance for historical consistency, if present
        if 'USDT' in st.session_state['all_balances']:
            st.session_state['balance'] = st.session_state['all_balances']['USDT']
        else:
            st.session_state['balance'] = 0.0  # No USDT found
            st.warning("USDT balance not found in your Binance Testnet account. Defaulting to 0.0 for main balance display.")

        st.toast("Balances updated successfully from Binance Testnet!", icon="✅")

    except BinanceAPIException as e:
        st.toast(f"Error fetching Binance Testnet balance: {e.message}. Check API keys.", icon="❌")
        # Fallback to simulated or previous balance if API call fails
        if 'balance' not in st.session_state:
            st.session_state['balance'] = 10000.0  # Initial fallback if nothing else is set
        st.warning(f"Using current USDT balance: ${st.session_state['balance']:,.2f} (may not be real-time).")
    except Exception as e:
        st.toast(f"An unexpected error occurred while fetching balance: {e}", icon="❌")
        if 'balance' not in st.session_state:
            st.session_state['balance'] = 10000.0  # Initial fallback
        st.warning(f"Using current USDT balance: ${st.session_state['balance']:,.2f} (may not be real-time).")

# --- Feature Preprocessing ---
def preprocess(df):
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA_20'] + 2 * rolling_std
    df['BB_lower'] = df['MA_20'] - 2 * rolling_std
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['Log_Returns'].rolling(window=60).std()
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)

    features = [
        'Close', 'Volume', 'MA_20', 'MA_50', 'EMA_20',
        'BB_upper', 'BB_lower', 'RSI', 'Volatility',
        'Close_lag_1', 'Close_lag_2'
    ]
    return df[features].tail(60)

# --- Load Scalers from .pkl files ---
@st.cache_resource
def load_scalers():
    try:
        feat_scaler = joblib.load("models/feature_scaler.pkl")
        tgt_scaler = joblib.load("models/target_scaler.pkl")
        print("Feature Scaler:")
        print("Min:", feat_scaler.data_min_)
        print("Max:", feat_scaler.data_max_)
        print("Scale:", feat_scaler.scale_)
        print("Min value used in scaling:", feat_scaler.min_)

        print("\nTarget Scaler:")
        print("Min:", tgt_scaler.data_min_)
        print("Max:", tgt_scaler.data_max_)
        print("Scale:", tgt_scaler.scale_)
        print("Min value used in scaling:", tgt_scaler.min_)
        return feat_scaler, tgt_scaler
    except FileNotFoundError:
        st.error("Scaler files (feature_scaler.pkl, target_scaler.pkl) not found. Please ensure they are in the 'models' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading scalers: {str(e)}")
        st.stop()

feat_scaler, tgt_scaler = load_scalers()

# --- Scaling and Rescaling Functions ---
def scale_features(X):
    feature_cols_order = [
        'Close', 'Volume', 'MA_20', 'MA_50', 'EMA_20',
        'BB_upper', 'BB_lower', 'RSI', 'Volatility',
        'Close_lag_1', 'Close_lag_2'
    ]
    X_df = pd.DataFrame(X, columns=feature_cols_order)
    X_scaled = feat_scaler.transform(X_df).astype('float32')
    return X_scaled

def rescale_prediction(y_scaled):
    y_rescaled = np.zeros(3)
    y_rescaled[:2] = tgt_scaler.inverse_transform([y_scaled[:2]])[0]
    y_rescaled[2] = torch.sigmoid(torch.tensor(y_scaled[2])).item()
    return y_rescaled

# --- Load Test Prediction/Ground Truth NPYS ---
@st.cache_resource
def load_test_data_npys():
    try:
        test_predictions_npy = np.load('models/test_predictions.npy')
        test_ground_truth_npy = np.load('models/test_ground_truth.npy')
        return test_predictions_npy, test_ground_truth_npy
    except FileNotFoundError:
        st.warning("Test .npy files (test_predictions.npy, test_ground_truth.npy) not found. Skipping loading. Please ensure they are in the 'models' directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading test .npy files: {str(e)}")
        return None, None

test_predictions_npy, test_ground_truth_npy = load_test_data_npys()

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
        loss = criterion(output[:,:2], y[:,:2])
        loss.backward()
        optimizer.step()
    model.eval()
    return f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | Training loss: {loss.item():.6f}"

# --- Load Model ---
@st.cache_resource
def load_model():
    model = MultiTaskTransformer(input_size=11, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    for m in model.modules():
        if isinstance(m, DyT):
            m.a.data.fill_(0.95)

    state_dict = torch.load("models/model_optuna_best.pth", map_location="cpu")
    #state_dict = torch.load("models/model_minutely_pytorch.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Notification System ---
def send_notification(message, signal_type):
    # Browser notification
    js = f"""
    <script>
        if ('Notification' in window && Notification.permission === 'granted') {{
            new Notification('{message}', {{
                body: 'Trading Signal',
                icon: 'https://cryptologos.cc/logos/bitcoin-btc-logo.png'
            }});
        }}
    </script>
    """
    components.html(js, height=0)

    # Streamlit toast notification
    if signal_type == 'buy' or signal_type == 'take_profit':
        st.toast(message, icon="📈")
    elif signal_type == 'sell' or signal_type == 'stop_loss':
        st.toast(message, icon="📉")
    else:
        st.toast(message)

# --- Streamlit UI Initialization ---
if 'predicted_history' not in st.session_state:
    st.session_state['predicted_history'] = []
if 'training_log' not in st.session_state:
    st.session_state['training_log'] = []
if 'trades' not in st.session_state:
    st.session_state['trades'] = []
if 'position' not in st.session_state:
    st.session_state['position'] = None
if 'balance' not in st.session_state:
    st.session_state['balance'] = 10000.0  # Initial simulated balance
if 'all_balances' not in st.session_state:
    st.session_state['all_balances'] = {}  # To store all fetched balances

# --- Title + Dropdown ---
st.title("Live Crypto Minutely Trading Dashboard")
selected_symbol = st.selectbox("Pick a trading pair:", TOP_COINS, index=0)

# Fetch balances once (or when balances dict is empty)
if not st.session_state['all_balances']:
    fetch_binance_balance()

# Display balances
st.markdown("### Current Binance Testnet Balances")
if st.session_state['all_balances']:
    balance_data = [{"Asset": k, "Free Balance": f"{v:,.8f}"}
                    for k, v in st.session_state['all_balances'].items()]
    st.dataframe(pd.DataFrame(balance_data), use_container_width=True)
else:
    st.info("No balances fetched yet or all balances are zero.")

# --- Sell-all button (kept original BTC logic) ---
col_buttons = st.columns(2)
with col_buttons[0]:
    if st.button("Sell All Coins", help="Liquidate all your holdings on Testnet."):
        base = selected_symbol.replace("USDT", "")
        bal = st.session_state['all_balances'].get(base, 0.0)
        if bal > 0:
            try:
                info = client.get_symbol_info(selected_symbol)
                lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 0.000001
                qty_prec = int(-np.log10(step_size)) if step_size < 1 else 0
                qty = np.floor(bal / step_size) * step_size
                qty = float(f"{qty:.{qty_prec}f}")
                if qty > 0:
                    order = client.create_order(symbol=selected_symbol, side='SELL',
                                                type='MARKET', quantity=qty)
                    st.toast(f"Sold {qty} {base} – {order['orderId']}", icon="✅")
                    st.session_state['position'] = None
                else:
                    st.toast("Nothing to sell or qty too small.", icon="ℹ️")
            except Exception as e:
                st.toast(str(e), icon="❌")
            fetch_binance_balance()
            st.rerun()
        else:
            st.toast(f"No {base} balance.", icon="ℹ️")

# --- Sidebar unchanged ---
st.sidebar.header("Trading Parameters")
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 5.0, 1.0) / 100
take_profit_pct = st.sidebar.slider("Take Profit (%)", 0.1, 5.0, 2.0) / 100
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0) / 100
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0.0, 100.0, 60.0) / 100

# --- Load model & data for the selected symbol ---
model = load_model()
raw_df = fetch_binance_ohlcv(symbol=selected_symbol)
X_raw = preprocess(raw_df)
X_scaled = scale_features(X_raw.values)

if X_raw.shape[0] >= 60:
    log_line = online_train(model, X_scaled)
    st.session_state['training_log'].append(log_line)
    if len(st.session_state['training_log']) % 10 == 0:
        torch.save(model.state_dict(), "model_minutely_full_new.pth")

# --- Display recent data ---
st.subheader(f"Recent {selected_symbol} Minutely Data")
st.dataframe(X_raw.tail(), use_container_width=True)

# --- Prediction block ---
st.markdown("### Model Prediction")
if X_raw.shape[0] < 60:
    st.warning("Not enough data to make a prediction (need at least 60 minutes).")
else:
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction_scaled = model(input_tensor).squeeze().numpy()
        prediction = rescale_prediction(prediction_scaled)
        close = prediction[0]
        vol = abs(prediction[1])
        direction_prob = prediction[2]
        direction = 1 if direction_prob > 0.5 else 0
        confidence = max(direction_prob, 1 - direction_prob)
        current_price = raw_df["Close"].iloc[-1]
        next_time = raw_df["Open Time"].iloc[-1] + pd.Timedelta(minutes=1)
        st.session_state['predicted_history'].append((next_time, close))

    # --- Trading logic (kept identical, only symbol changed) ---
    account_usdt_balance = st.session_state['all_balances'].get('USDT', 0.0)
    st.session_state['balance'] = account_usdt_balance
    risk_amount = account_usdt_balance * risk_per_trade

    min_notional = 10.0
    quantity_precision = 6
    step_size = 0.000001
    try:
        info = client.get_symbol_info(selected_symbol)
        min_notional_filter = next((f for f in info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
        if min_notional_filter:
            min_notional = float(min_notional_filter['minNotional'])
        lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            quantity_precision = int(-np.log10(step_size)) if step_size < 1 else 0
    except Exception as e:
        st.toast(f"Could not fetch symbol info: {e}", icon="❌")

    position_size = min(risk_amount / (stop_loss_pct * current_price),
                        account_usdt_balance / current_price)
    min_qty = min_notional / current_price
    position_size = max(position_size, min_qty)
    position_size = np.floor(position_size / step_size) * step_size
    position_size = float(f"{position_size:.{quantity_precision}f}")

    if position_size * current_price < min_notional * 0.999:
        position_size = 0.0

    # --- BUY logic ---
    if direction == 1 and st.session_state['position'] is None and confidence >= confidence_threshold:
        if position_size > 0:
            try:
                order = client.create_order(symbol=selected_symbol, side='BUY', type='MARKET',
                                            quantity=f"{position_size:.{quantity_precision}f}")
                executed_qty = float(order.get('executedQty', position_size))
                avg_price = float(order['fills'][0]['price']) if order.get('fills') else current_price
                st.session_state['position'] = {
                    'type': 'long',
                    'entry_price': avg_price,
                    'size': executed_qty,
                    'stop_loss': avg_price * (1 - stop_loss_pct),
                    'take_profit': avg_price * (1 + take_profit_pct)
                }
                send_notification(f"Buy {selected_symbol} @ {avg_price:,.2f}", 'buy')
                fetch_binance_balance()
                st.rerun()
            except Exception as e:
                st.toast(str(e), icon="❌")
        else:
            st.toast("Buy signal but size too small.", icon="🚫")

    # --- SELL logic (short) ---
    elif direction == 0 and st.session_state['position'] is None and confidence >= confidence_threshold:
        base = selected_symbol.replace("USDT", "")
        avail_base = st.session_state['all_balances'].get(base, 0.0)
        if avail_base < position_size:
            st.toast(f"Not enough {base} to sell short.", icon="🚫")
        elif position_size > 0:
            try:
                order = client.create_order(symbol=selected_symbol, side='SELL', type='MARKET',
                                            quantity=f"{position_size:.{quantity_precision}f}")
                executed_qty = float(order.get('executedQty', position_size))
                avg_price = float(order['fills'][0]['price']) if order.get('fills') else current_price
                st.session_state['position'] = {
                    'type': 'short',
                    'entry_price': avg_price,
                    'size': executed_qty,
                    'stop_loss': avg_price * (1 + stop_loss_pct),
                    'take_profit': avg_price * (1 - take_profit_pct)
                }
                send_notification(f"Sell {selected_symbol} @ {avg_price:,.2f}", 'sell')
                fetch_binance_balance()
                st.rerun()
            except Exception as e:
                st.toast(str(e), icon="❌")
        else:
            st.toast("Sell signal but size too small.", icon="🚫")

    # --- Exit logic (unchanged except symbol) ---
    if st.session_state['position']:
        pos = st.session_state['position']
        info = client.get_symbol_info(selected_symbol)
        lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        qty_prec_close = 6
        if lot_size_filter:
            qty_prec_close = int(-np.log10(float(lot_size_filter['stepSize']))) if float(lot_size_filter['stepSize']) < 1 else 0

        # LONG exit
        if pos['type'] == 'long':
            if current_price <= pos['stop_loss']:
                try:
                    order = client.create_order(symbol=selected_symbol, side='SELL', type='MARKET',
                                                quantity=f"{pos['size']:.{qty_prec_close}f}")
                    pnl = (current_price - pos['entry_price']) * pos['size']
                    st.session_state['trades'].append({
                        'time': pd.Timestamp.now(),
                        'pnl': pnl,
                        'type': 'long',
                        'exit_reason': 'stop_loss',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'size': pos['size']
                    })
                    st.session_state['balance'] += pnl
                    send_notification("Stop loss hit", 'stop_loss')
                    st.session_state['position'] = None
                    fetch_binance_balance()
                    st.rerun()
                except Exception as e:
                    st.toast(str(e), icon="❌")
            elif current_price >= pos['take_profit']:
                try:
                    order = client.create_order(symbol=selected_symbol, side='SELL', type='MARKET',
                                                quantity=f"{pos['size']:.{qty_prec_close}f}")
                    pnl = (current_price - pos['entry_price']) * pos['size']
                    st.session_state['trades'].append({
                        'time': pd.Timestamp.now(),
                        'pnl': pnl,
                        'type': 'long',
                        'exit_reason': 'take_profit',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'size': pos['size']
                    })
                    st.session_state['balance'] += pnl
                    send_notification("Take profit hit", 'take_profit')
                    st.session_state['position'] = None
                    fetch_binance_balance()
                    st.rerun()
                except Exception as e:
                    st.toast(str(e), icon="❌")

        # SHORT exit
        else:
            if current_price >= pos['stop_loss']:
                try:
                    order = client.create_order(symbol=selected_symbol, side='BUY', type='MARKET',
                                                quantity=f"{pos['size']:.{qty_prec_close}f}")
                    pnl = (pos['entry_price'] - current_price) * pos['size']
                    st.session_state['trades'].append({
                        'time': pd.Timestamp.now(),
                        'pnl': pnl,
                        'type': 'short',
                        'exit_reason': 'stop_loss',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'size': pos['size']
                    })
                    st.session_state['balance'] += pnl
                    send_notification("Stop loss hit", 'stop_loss')
                    st.session_state['position'] = None
                    fetch_binance_balance()
                    st.rerun()
                except Exception as e:
                    st.toast(str(e), icon="❌")
            elif current_price <= pos['take_profit']:
                try:
                    order = client.create_order(symbol=selected_symbol, side='BUY', type='MARKET',
                                                quantity=f"{pos['size']:.{qty_prec_close}f}")
                    pnl = (pos['entry_price'] - current_price) * pos['size']
                    st.session_state['trades'].append({
                        'time': pd.Timestamp.now(),
                        'pnl': pnl,
                        'type': 'short',
                        'exit_reason': 'take_profit',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'size': pos['size']
                    })
                    st.session_state['balance'] += pnl
                    send_notification("Take profit hit", 'take_profit')
                    st.session_state['position'] = None
                    fetch_binance_balance()
                    st.rerun()
                except Exception as e:
                    st.toast(str(e), icon="❌")

# --- Metrics & Viz unchanged ---
sharpe, total_pnl, win_rate = calculate_metrics(st.session_state['trades'])
st.markdown("### Trading Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("Total P/L (USDT)", f"${total_pnl:,.2f}")
col3.metric("Win Rate (%)", f"{win_rate:.1f}%")
col4.metric("USDT Account Balance", f"${st.session_state['balance']:,.2f}")

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

st.markdown(f"### {selected_symbol} – Actual vs Predicted Close Price (Auto-refreshes every 60s)")
timestamps = raw_df["Open Time"].tail(60).tolist()
actual_close = raw_df["Close"].tail(60).tolist()
next_time = timestamps[-1] + pd.Timedelta(minutes=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=timestamps, y=actual_close, mode="lines+markers", name="Actual Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=[next_time], y=[close], mode="markers+text", marker=dict(color="red", size=10), name="Next Prediction"))
if len(st.session_state['predicted_history']) > 1:
    pred_times, pred_closes = zip(*st.session_state['predicted_history'])
    fig.add_trace(go.Scatter(x=list(pred_times), y=list(pred_closes), mode="lines+markers",
                             line=dict(color="red", width=2, dash="dash"), name="Predicted Line"))
fig.update_layout(title=f"{selected_symbol}", height=500, margin=dict(l=20, r=20, t=30, b=20),
                  showlegend=True, xaxis_title="Time", yaxis_title="Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Trade History")
if st.session_state['trades']:
    trade_df = pd.DataFrame(st.session_state['trades'])
    st.dataframe(trade_df[['time', 'type', 'entry_price', 'exit_price', 'size', 'pnl', 'exit_reason']], use_container_width=True)
else:
    st.info("No trades executed yet.")

st.markdown("### 🔧 Online Training Log")
with st.expander("Show training updates", expanded=False):
    if st.session_state['training_log']:
        for log in reversed(st.session_state['training_log'][-10:]):
            st.markdown(f"- {log}")
    else:
        st.info("No training events yet.")

# --- Debug .npy display ---
st.markdown("### Loaded Test Data (from .npy files)")
if test_predictions_npy is not None and test_ground_truth_npy is not None:
    st.write("Shape of `test_predictions.npy`:", test_predictions_npy.shape)
    st.dataframe(pd.DataFrame(test_predictions_npy[:5], columns=['Pred Close', 'Pred Vol', 'Pred Dir']))
    st.write("Shape of `test_ground_truth.npy`:", test_ground_truth_npy.shape)
    st.dataframe(pd.DataFrame(test_ground_truth_npy[:5], columns=['Actual Close', 'Actual Vol', 'Actual Dir']))
else:
    st.info("`test_predictions.npy` or `test_ground_truth.npy` were not found or could not be loaded.")

st_autorefresh(interval=60000, key="refresh")

components.html("""
<script>
if ('Notification' in window && Notification.permission !== 'granted') {
    Notification.requestPermission();
}
</script>
""", height=0)