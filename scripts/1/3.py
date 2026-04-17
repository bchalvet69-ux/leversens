import numpy as np
import pandas as pd
import par84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _helpers import fetch, FREQ_CONFIGS, ticker_safe, base_layout, get_rangebreaks, run_backtest, plot_backtest

# ============================================================================
# Script 3 — MACD
# Dashboard (3-1.html): BTC-USD, 6-month daily, oscillator + signals
# Analytics:  5 assets × 2 frequencies → 3-{freq}-{TICKER}.html
# ============================================================================

BULL = par84.BULL
BEAR = par84.BEAR


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd   = ema_fast - ema_slow
    sig    = macd.ewm(span=signal, adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist


def plot_macd(df, title, fname, subtitle=""):
    macd, sig, hist = compute_macd(df['Close'])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)

    # Price (row 1) — candles
    fig.add_candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=BULL, width=1), fillcolor=BULL),
        decreasing=dict(line=dict(color=BEAR, width=1), fillcolor=BEAR),
        showlegend=False, name='Price', row=1, col=1,
    )

    # Signals based on MACD/Signal crossovers
    cross_up   = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))

    fig.add_scatter(
        x=df.index[cross_up], y=df['Close'][cross_up],
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                                    line=dict(color='#fff', width=1)),
        name='Long', row=1, col=1,
    )
    fig.add_scatter(
        x=df.index[cross_down], y=df['Close'][cross_down],
        mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                                    line=dict(color='#fff', width=1)),
        name='Short', row=1, col=1,
    )

    # MACD histogram + lines (row 2)
    bar_colors = np.where(hist >= 0, 'rgba(34,197,94,0.6)', 'rgba(239,68,68,0.55)')
    fig.add_bar(x=df.index, y=hist, marker=dict(color=bar_colors, line=dict(width=0)),
                showlegend=False, name='Hist', row=2, col=1)
    fig.add_scatter(x=df.index, y=macd, mode='lines',
                    line=dict(color='#f59e0b', width=1.6), name='MACD', row=2, col=1)
    fig.add_scatter(x=df.index, y=sig, mode='lines',
                    line=dict(color='#a78bfa', width=1.3, dash='dot'), name='Signal', row=2, col=1)

    rb = get_rangebreaks(df)
    fig.update_layout(**base_layout(title, subtitle, rangebreaks=rb))
    fig.update_yaxes(title='', row=1, col=1, side='right', gridcolor='rgba(59,130,246,0.06)')
    fig.update_yaxes(title='MACD', row=2, col=1, side='right',
                     gridcolor='rgba(59,130,246,0.06)', tickfont=dict(size=9, color='#475569'))

    fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ {fname}")


# ═══ Dashboard: BTC-USD 6-month ═══
print("▶ MACD · Dashboard (3-1.html): BTC-USD 6-month")
df = fetch('BTC-USD', '6mo', '1d')
if df is not None:
    plot_macd(df, 'MACD', '3-1.html', 'BTC-USD · Bitcoin · Daily · 6m')


# ═══ Analytics: per asset × 2 frequencies ═══
print("\n▶ MACD · Analytics (asset × freq)")
for ticker in par84.TICKERS:
    safe = ticker_safe(ticker)
    tname = par84.TICKER_NAMES[ticker]
    for key, cfg in FREQ_CONFIGS.items():
        df = fetch(ticker, cfg['period'], cfg['interval'])
        if df is None:
            continue
        fname = f"3-{key}-{safe}.html"
        plot_macd(df, 'MACD', fname, f'{ticker} · {tname} · {cfg["label"]}')

        # Backtest
        macd, sig, hist = compute_macd(df['Close'])
        buy_sig  = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        sell_sig = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        bt_df, bt_stats = run_backtest(df, buy_sig, sell_sig)
        bt_fname = f"3-bt-{key}-{safe}.html"
        plot_backtest(bt_df, bt_stats, 'MACD', bt_fname, f'{ticker} · {tname} · {cfg["label"]}')

print("\n✓ Script 3 done")
