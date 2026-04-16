import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import par84

# ============================================================================
# Script 7 - BTC Price vs On-Chain Metrics (New Addresses & Transactions)
#            + Buy/Sell signals based on LARGE on-chain movements only
#            Uses Bollinger-style bands (MA ± 2σ) to detect outlier activity
#            Uses blockchain.com free API (no key required)
# ============================================================================

images = par84.img_path + '7-'

# ── 1. BTC Price from yfinance ──
print("Downloading BTC price data...")
btc = yf.Ticker('BTC-USD').history(period='2y', interval='1d', auto_adjust=True)
btc_price = btc['Close'].dropna()
if btc_price.index.tz is not None:
    btc_price.index = btc_price.index.tz_localize(None)
print(f"  ✓ BTC price: {len(btc_price)} days")

# ── 2. On-chain data from blockchain.com API ──
def fetch_blockchain_chart(chart_name, timespan='2years'):
    """Fetch data from blockchain.com charts API."""
    url = f"https://api.blockchain.info/charts/{chart_name}"
    params = {'timespan': timespan, 'format': 'json', 'sampled': 'true'}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        values = data.get('values', [])
        df = pd.DataFrame(values)
        df['x'] = pd.to_datetime(df['x'], unit='s')
        df = df.set_index('x')
        df.index.name = 'Date'
        df.columns = [chart_name]
        return df
    except Exception as e:
        print(f"  ✗ Error fetching {chart_name}: {e}")
        return pd.DataFrame()

print("Fetching on-chain data from blockchain.com...")
df_addresses = fetch_blockchain_chart('n-unique-addresses')
df_transactions = fetch_blockchain_chart('n-transactions')
print(f"  ✓ Unique addresses: {len(df_addresses)} points")
print(f"  ✓ Transactions: {len(df_transactions)} points")

# ── 3. Align all data ──
price_df = btc_price.to_frame('BTC_Price')

if not df_addresses.empty:
    df_addresses = df_addresses.resample('D').mean().ffill()
if not df_transactions.empty:
    df_transactions = df_transactions.resample('D').mean().ffill()

merged = price_df.join(df_addresses, how='left').join(df_transactions, how='left').dropna()
print(f"  ✓ Merged dataset: {len(merged)} rows")

# ── 4. Generate buy/sell signals based on LARGE movements ──
# Strategy: Bollinger-style bands on on-chain metrics (MA ± 2σ)
# Signal only when BOTH metrics exceed their upper band (= abnormally high activity = BUY)
# or BOTH drop below lower band (= abnormally low activity = SELL)
LOOKBACK = 30
N_STD = 1.5

addr_ma = merged['n-unique-addresses'].rolling(LOOKBACK).mean()
addr_std = merged['n-unique-addresses'].rolling(LOOKBACK).std()
addr_upper = addr_ma + N_STD * addr_std
addr_lower = addr_ma - N_STD * addr_std

tx_ma = merged['n-transactions'].rolling(LOOKBACK).mean()
tx_std = merged['n-transactions'].rolling(LOOKBACK).std()
tx_upper = tx_ma + N_STD * tx_std
tx_lower = tx_ma - N_STD * tx_std

# Buy: BOTH metrics break above their upper band (strong network surge)
addr_above = merged['n-unique-addresses'] > addr_upper
tx_above = merged['n-transactions'] > tx_upper
buy_mask = addr_above & tx_above

# Sell: BOTH metrics break below their lower band (strong network contraction)
addr_below = merged['n-unique-addresses'] < addr_lower
tx_below = merged['n-transactions'] < tx_lower
sell_mask = addr_below & tx_below

# Only keep signal changes
buy_signal  = buy_mask & (~buy_mask.shift(1).fillna(False))
sell_signal = sell_mask & (~sell_mask.shift(1).fillna(False))

# Reduce noise: minimum 14 days between signals (more selective)
def filter_signals(signal_series, min_gap=14):
    filtered = signal_series.copy()
    last_idx = None
    for i, (idx, val) in enumerate(filtered.items()):
        if val:
            if last_idx is not None and (idx - last_idx).days < min_gap:
                filtered.iloc[i] = False
            else:
                last_idx = idx
    return filtered

buy_signal  = filter_signals(buy_signal, min_gap=14)
sell_signal = filter_signals(sell_signal, min_gap=14)

buy_dates  = merged.index[buy_signal]
sell_dates = merged.index[sell_signal]

print(f"  ✓ Buy signals: {len(buy_dates)}, Sell signals: {len(sell_dates)}")
print(f"    (using Bollinger bands: MA({LOOKBACK}) ± {N_STD}σ, min gap 14d)")

# ── 5. DASHBOARD CHART (7-1.html): Compact view for 3×2 grid ──
fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# BTC Price
fig1.add_trace(
    go.Scatter(
        x=merged.index, y=merged['BTC_Price'],
        name='BTC Price',
        line=dict(color='#f59e0b', width=2),
        hovertemplate="$%{y:,.0f}<extra>BTC Price</extra>",
    ),
    secondary_y=False,
)

# New Addresses
fig1.add_trace(
    go.Scatter(
        x=merged.index, y=merged['n-unique-addresses'],
        name='Addresses',
        line=dict(color='#a78bfa', width=1.2),
        opacity=0.7,
        hovertemplate="%{y:,.0f}<extra>Addresses</extra>",
    ),
    secondary_y=True,
)

# Transactions
fig1.add_trace(
    go.Scatter(
        x=merged.index, y=merged['n-transactions'],
        name='Transactions',
        line=dict(color='#34d399', width=1.2, dash='dot'),
        opacity=0.7,
        hovertemplate="%{y:,.0f}<extra>Transactions</extra>",
    ),
    secondary_y=True,
)

# Buy triangles
if len(buy_dates) > 0:
    fig1.add_trace(
        go.Scatter(
            x=buy_dates,
            y=merged.loc[buy_dates, 'BTC_Price'],
            mode='markers',
            name='Buy',
            marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                        line=dict(width=1, color='#16a34a')),
            hovertemplate="BUY $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

# Sell triangles
if len(sell_dates) > 0:
    fig1.add_trace(
        go.Scatter(
            x=sell_dates,
            y=merged.loc[sell_dates, 'BTC_Price'],
            mode='markers',
            name='Sell',
            marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                        line=dict(width=1, color='#dc2626')),
            hovertemplate="SELL $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

fig1.update_layout(
    template='plotly_dark',
    paper_bgcolor=par84.CHART_BG,
    plot_bgcolor=par84.CHART_BG,
    title=par84.dash_title('BTC · ON-CHAIN'),
    margin=dict(l=10, r=10, t=52, b=30),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5,
        font=dict(size=9),
    ),
    font=dict(color='#cbd5e1'),
    hovermode='x unified',
)
fig1.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
fig1.update_yaxes(
    title_text='', secondary_y=False,
    gridcolor='rgba(255,255,255,0.05)',
    tickformat='$,.0f',
)
fig1.update_yaxes(
    title_text='', secondary_y=True,
    gridcolor='rgba(255,255,255,0.03)',
    tickformat=',.0f',
)

fig1.write_html(images + '1.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 7-1.html: BTC On-Chain dashboard chart")

# ── 6. ANALYTICS CHART (7-2.html): Full-size with Bollinger bands on metrics ──
fig2 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.04,
    subplot_titles=['BTC Price + Buy/Sell Signals', 'Unique Addresses / Day (+ Bollinger Band)', 'Transactions / Day (+ Bollinger Band)'],
)

# Row 1: BTC Price + signals
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=merged['BTC_Price'],
        name='BTC Price',
        line=dict(color='#f59e0b', width=2),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.06)',
        hovertemplate="$%{y:,.0f}<extra>BTC Price</extra>",
    ),
    row=1, col=1,
)

if len(buy_dates) > 0:
    fig2.add_trace(
        go.Scatter(
            x=buy_dates,
            y=merged.loc[buy_dates, 'BTC_Price'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=14, color='#22c55e',
                        line=dict(width=1.5, color='#16a34a')),
            hovertemplate="BUY $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

if len(sell_dates) > 0:
    fig2.add_trace(
        go.Scatter(
            x=sell_dates,
            y=merged.loc[sell_dates, 'BTC_Price'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=14, color='#ef4444',
                        line=dict(width=1.5, color='#dc2626')),
            hovertemplate="SELL $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

# Row 2: Unique Addresses + Bollinger bands
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=merged['n-unique-addresses'],
        name='Unique Addresses',
        line=dict(color='#a78bfa', width=1.5),
        hovertemplate="%{y:,.0f}<extra>Addresses</extra>",
    ),
    row=2, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=addr_upper,
        name=f'Upper Band ({N_STD}σ)',
        line=dict(color='rgba(167,139,250,0.5)', width=1, dash='dash'),
        showlegend=False,
    ),
    row=2, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=addr_lower,
        name=f'Lower Band ({N_STD}σ)',
        line=dict(color='rgba(167,139,250,0.5)', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(167,139,250,0.06)',
        showlegend=False,
    ),
    row=2, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=addr_ma,
        name=f'Addresses MA({LOOKBACK})',
        line=dict(color='#a78bfa', width=1, dash='dot'),
        opacity=0.5,
        showlegend=False,
    ),
    row=2, col=1,
)

# Row 3: Transactions + Bollinger bands
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=merged['n-transactions'],
        name='Transactions',
        line=dict(color='#34d399', width=1.5),
        hovertemplate="%{y:,.0f}<extra>Transactions</extra>",
    ),
    row=3, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=tx_upper,
        name=f'Upper Band ({N_STD}σ)',
        line=dict(color='rgba(52,211,153,0.5)', width=1, dash='dash'),
        showlegend=False,
    ),
    row=3, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=tx_lower,
        name=f'Lower Band ({N_STD}σ)',
        line=dict(color='rgba(52,211,153,0.5)', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(52,211,153,0.06)',
        showlegend=False,
    ),
    row=3, col=1,
)
fig2.add_trace(
    go.Scatter(
        x=merged.index, y=tx_ma,
        name=f'Transactions MA({LOOKBACK})',
        line=dict(color='#34d399', width=1, dash='dot'),
        opacity=0.5,
        showlegend=False,
    ),
    row=3, col=1,
)

fig2.update_layout(
    template='plotly_dark',
    paper_bgcolor=par84.CHART_BG,
    plot_bgcolor=par84.CHART_BG,
    title=par84.dash_title('BTC · ON-CHAIN ANALYSIS'),
    margin=dict(l=60, r=20, t=60, b=30),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        orientation='h', yanchor='top', y=-0.06, xanchor='center', x=0.5,
    ),
    font=dict(color='#cbd5e1'),
    hovermode='x unified',
    height=700,
)

# Style subplot titles
for ann in fig2['layout']['annotations']:
    ann['font'] = dict(size=11, color='#60a5fa')

for row in [1, 2, 3]:
    fig2.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=row, col=1)
    fig2.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=row, col=1)

fig2.update_yaxes(tickformat='$,.0f', row=1, col=1)
fig2.update_yaxes(tickformat=',.0f', row=2, col=1)
fig2.update_yaxes(tickformat=',.0f', row=3, col=1)

fig2.write_html(images + '2.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 7-2.html: BTC On-Chain analytics chart")

# ── 7. Stats ──
if len(merged) > 10:
    corr_addr = merged['BTC_Price'].corr(merged['n-unique-addresses'])
    corr_tx = merged['BTC_Price'].corr(merged['n-transactions'])
    print(f"\nCorrelation BTC Price vs Unique Addresses: {corr_addr:.4f}")
    print(f"Correlation BTC Price vs Transactions:      {corr_tx:.4f}")

print(f"\n✓ Script 7 (BTC On-Chain) completed successfully!")
