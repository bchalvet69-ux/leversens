import numpy as np
import pandas as pd
import par84
import plotly.graph_objects as go
from _helpers import fetch, FREQ_CONFIGS, ticker_safe, base_layout, get_rangebreaks, run_backtest, plot_backtest

# ============================================================================
# Script 5 — Bollinger Bands
# Dashboard (5-1.html): GOLD, 1-year daily — SMA20 ± 2σ + buy/sell
# Analytics: 5 assets × 2 freq → 5-{freq}-{TICKER}.html
# ============================================================================

BULL = par84.BULL
BEAR = par84.BEAR


def compute_bollinger(close, window=20, n_std=2):
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower


def plot_bollinger(df, title, fname, subtitle=""):
    ma, upper, lower = compute_bollinger(df['Close'])

    # Signals — close crosses back above lower = buy, crosses back below upper = sell
    close = df['Close']
    buy  = (close.shift(1) < lower.shift(1)) & (close >= lower)
    sell = (close.shift(1) > upper.shift(1)) & (close <= upper)

    fig = go.Figure(layout=par84.layout)

    # Bands (purple)
    fig.add_scatter(x=df.index, y=upper, mode='lines',
                    line=dict(color='rgba(167,139,250,0.7)', width=1),
                    name='Upper')
    fig.add_scatter(x=df.index, y=lower, mode='lines',
                    line=dict(color='rgba(167,139,250,0.7)', width=1),
                    fill='tonexty', fillcolor='rgba(167,139,250,0.07)',
                    name='Lower')
    fig.add_scatter(x=df.index, y=ma, mode='lines',
                    line=dict(color='#f59e0b', width=1.4, dash='dot'),
                    name='SMA 20')

    # Candles
    fig.add_candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=BULL, width=1), fillcolor=BULL),
        decreasing=dict(line=dict(color=BEAR, width=1), fillcolor=BEAR),
        showlegend=False, name='Price',
    )

    # Signals
    fig.add_scatter(x=df.index[buy], y=close[buy], mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                                line=dict(color='#fff', width=1)),
                    name='Buy')
    fig.add_scatter(x=df.index[sell], y=close[sell], mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                                line=dict(color='#fff', width=1)),
                    name='Sell')

    rb = get_rangebreaks(df)
    fig.update_layout(**base_layout(title, subtitle, rangebreaks=rb))
    fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ {fname}")


# ═══ Dashboard: GOLD 1-year ═══
print("▶ Bollinger · Dashboard (5-1.html): GOLD 1-year")
df = fetch('GOLD', '1y', '1d')
if df is not None:
    plot_bollinger(df, 'BOLLINGER BANDS', '5-1.html', 'GOLD · Barrick Gold · Daily · 1y')


# ═══ Analytics: per asset × 2 frequencies ═══
print("\n▶ Bollinger · Analytics (asset × freq)")
for ticker in par84.TICKERS:
    safe = ticker_safe(ticker)
    tname = par84.TICKER_NAMES[ticker]
    for key, cfg in FREQ_CONFIGS.items():
        df = fetch(ticker, cfg['period'], cfg['interval'])
        if df is None:
            continue
        fname = f"5-{key}-{safe}.html"
        plot_bollinger(df, 'BOLLINGER BANDS', fname, f'{ticker} · {tname} · {cfg["label"]}')

        # Backtest
        ma, upper, lower = compute_bollinger(df['Close'])
        close = df['Close']
        buy_sig  = (close.shift(1) < lower.shift(1)) & (close >= lower)
        sell_sig = (close.shift(1) > upper.shift(1)) & (close <= upper)
        bt_df, bt_stats = run_backtest(df, buy_sig, sell_sig)
        bt_fname = f"5-bt-{key}-{safe}.html"
        plot_backtest(bt_df, bt_stats, 'Bollinger', bt_fname, f'{ticker} · {tname} · {cfg["label"]}')

print("\n✓ Script 5 done")
