import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="COT Signal Engine", layout="wide")
st.title("COT Positioning & Momentum Dashboard")
st.markdown("Filtering Commitments of Traders data for actionable macroeconomic signals.")

# --- 2. DATA SIMULATION (The Pipeline) ---
# In production, replace this function with a script that downloads the CFTC weekly CSV.
@st.cache_data
def load_mock_cot_data():
    dates = pd.date_range(end=datetime.today(), periods=156, freq='W-TUE') # 3 Years of Tuesdays
    assets = ['Gold', 'EUR/USD', 'Crude Oil', 'S&P 500', '10Y T-Note']
    
    data = []
    for asset in assets:
        # Simulate base metrics
        oi = np.random.randint(200000, 600000, size=len(dates))
        nc_long = oi * np.random.uniform(0.1, 0.4, size=len(dates))
        nc_short = oi * np.random.uniform(0.1, 0.4, size=len(dates))
        
        # Add artificial trends to create extremes
        if asset == 'Gold':
            nc_long[-10:] = nc_long[-10:] * 1.5 # Spike recent longs
        
        df = pd.DataFrame({
            'Date': dates,
            'Asset': asset,
            'Open_Interest': oi,
            'NC_Long': nc_long,
            'NC_Short': nc_short,
            'Price': np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))
        })
        data.append(df)
        
    return pd.concat(data)

df = load_mock_cot_data()

# --- 3. QUANTITATIVE LOGIC (The Math) ---
def compute_signals(data):
    # Calculate Raw Net and Normalize by Open Interest
    data['NC_Net'] = data['NC_Long'] - data['NC_Short']
    data['Net_OI_Ratio'] = data['NC_Net'] / data['Open_Interest']
    
    # Calculate 3-Year Rolling Metrics
    data['3Y_Mean'] = data.groupby('Asset')['Net_OI_Ratio'].transform(lambda x: x.rolling(156, min_periods=52).mean())
    data['3Y_Std'] = data.groupby('Asset')['Net_OI_Ratio'].transform(lambda x: x.rolling(156, min_periods=52).std())
    data['3Y_Max'] = data.groupby('Asset')['Net_OI_Ratio'].transform(lambda x: x.rolling(156, min_periods=52).max())
    data['3Y_Min'] = data.groupby('Asset')['Net_OI_Ratio'].transform(lambda x: x.rolling(156, min_periods=52).min())
    
    # Z-Score and Percentile Rank
    data['Z_Score'] = (data['Net_OI_Ratio'] - data['3Y_Mean']) / data['3Y_Std']
    data['Percentile'] = ((data['Net_OI_Ratio'] - data['3Y_Min']) / (data['3Y_Max'] - data['3Y_Min'])) * 100
    
    # Price Momentum (Simple 4-Week Rate of Change)
    data['Momentum_4W'] = data.groupby('Asset')['Price'].transform(lambda x: x.pct_change(4))
    
    return data.dropna()

processed_df = compute_signals(df)

# --- 4. DASHBOARD UI ---
st.sidebar.header("Signal Filters")
upper_extreme = st.sidebar.slider("Bullish Extreme Percentile", 70, 100, 90)
lower_extreme = st.sidebar.slider("Bearish Extreme Percentile", 0, 30, 10)
momentum_filter = st.sidebar.checkbox("Require Momentum Confirmation?", value=True)

# Get most recent week's data
latest_date = processed_df['Date'].max()
current_data = processed_df[processed_df['Date'] == latest_date].copy()

# Generate Signals based on user rules
def generate_signal(row):
    is_bull_extreme = row['Percentile'] >= upper_extreme
    is_bear_extreme = row['Percentile'] <= lower_extreme
    
    if momentum_filter:
        if is_bear_extreme and row['Momentum_4W'] > 0: return "🟢 BUY (Reversal)"
        if is_bull_extreme and row['Momentum_4W'] < 0: return "🔴 SELL (Reversal)"
        return "⚪ Neutral / Waiting"
    else:
        if is_bear_extreme: return "⚠️ Oversold Extreme"
        if is_bull_extreme: return "⚠️ Overbought Extreme"
        return "⚪ Neutral"

current_data['Signal'] = current_data.apply(generate_signal, axis=1)

# Display Matrix
st.subheader(f"Cross-Asset Positioning (As of {latest_date.strftime('%Y-%m-%d')})")
display_cols = ['Asset', 'Percentile', 'Z_Score', 'Momentum_4W', 'Signal']
styled_df = current_data[display_cols].style.format({
    'Percentile': '{:.1f}%',
    'Z_Score': '{:.2f}',
    'Momentum_4W': '{:.2%}'
}).map(lambda x: 'color: green' if 'BUY' in str(x) else ('color: red' if 'SELL' in str(x) else ''), subset=['Signal'])

st.dataframe(styled_df, use_container_width=True)

# --- 5. VISUALIZATION ---
st.subheader("Deep Dive: Historical Positioning")
selected_asset = st.selectbox("Select Asset to view history:", current_data['Asset'].unique())
asset_history = processed_df[processed_df['Asset'] == selected_asset]

fig = go.Figure()
# Plot Normalized Position
fig.add_trace(go.Scatter(x=asset_history['Date'], y=asset_history['Percentile'], name='3Y Percentile', line=dict(color='blue')))
# Add Extreme Threshold Lines
fig.add_hline(y=upper_extreme, line_dash="dot", line_color="red", annotation_text="Overbought")
fig.add_hline(y=lower_extreme, line_dash="dot", line_color="green", annotation_text="Oversold")

fig.update_layout(height=400, title=f"{selected_asset} Speculator Positioning vs Thresholds")
st.plotly_chart(fig, use_container_width=True)
