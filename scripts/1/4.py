import numpy as np
import pandas as pd
import par84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _helpers import fetch, FREQ_CONFIGS, ticker_safe, base_layout, get_rangebreaks, run_backtest, plot_backtest

# ============================================================================
# Script 4 — RSI
# Dashboard (4-1.html): CL=F, 6-month daily — price + RSI + buy/sell markers
# Analytics: 5 assets × 2 frequencies → 4-{freq}-{TICKER}.html
# ============================================================================

BULL = par84.BULL
BEAR = par84.BEAR


def compute_rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing (SMMA)
    roll_up   = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs  = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def plot_rsi(df, title, fname, subtitle=""):
    rsi = compute_rsi(df['Close'])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)

    fig.add_candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=BULL, width=1), fillcolor=BULL),
        decreasing=dict(line=dict(color=BEAR, width=1), fillcolor=BEAR),
        showlegend=False, name='Price', row=1, col=1,
    )

    # Entry when RSI crosses back up from <30, exit when crosses back down from >70
    rsi_prev = rsi.shift(1)
    long_entry  = (rsi_prev < 30) & (rsi >= 30)
    short_entry = (rsi_prev > 70) & (rsi <= 70)

    fig.add_scatter(
        x=df.index[long_entry], y=df['Close'][long_entry],
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                                    line=dict(color='#fff', width=1)),
        name='Long', row=1, col=1,
    )
    fig.add_scatter(
        x=df.index[short_entry], y=df['Close'][short_entry],
        mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                                    line=dict(color='#fff', width=1)),
        name='Short', row=1, col=1,
    )

    # RSI pane
    fig.add_scatter(x=df.index, y=rsi, mode='lines',
                    line=dict(color='#f59e0b', width=1.5), name='RSI (14)', row=2, col=1)
    # Overbought / oversold fill band
    fig.add_scatter(x=df.index, y=[70]*len(df), mode='lines',
                    line=dict(color='rgba(239,68,68,0.7)', width=1, dash='dash'),
                    showlegend=False, name='70', row=2, col=1)
    fig.add_scatter(x=df.index, y=[30]*len(df), mode='lines',
                    line=dict(color='rgba(34,197,94,0.7)', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(167,139,250,0.06)',
                    showlegend=False, name='30', row=2, col=1)
    # Midline
    fig.add_scatter(x=df.index, y=[50]*len(df), mode='lines',
                    line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'),
                    showlegend=False, name='50', row=2, col=1)

    rb = get_rangebreaks(df)
    fig.update_layout(**base_layout(title, subtitle, rangebreaks=rb))
    fig.update_yaxes(title='', row=1, col=1, side='right', gridcolor='rgba(59,130,246,0.06)')
    fig.update_yaxes(title='RSI', row=2, col=1, side='right',
                     range=[0, 100],
                     gridcolor='rgba(59,130,246,0.06)', tickfont=dict(size=9, color='#475569'))

    fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ {fname}")


# ═══ Dashboard: CL=F 6-month ═══
print("▶ RSI · Dashboard (4-1.html): CL=F 6-month")
df = fetch('CL=F', '6mo', '1d')
if df is not None:
    plot_rsi(df, 'RSI', '4-1.html', 'CL=F · Crude Oil · Daily · 6m')


# ═══ Analytics: per asset × 2 frequencies ═══
print("\n▶ RSI · Analytics (asset × freq)")
for ticker in par84.TICKERS:
    safe = ticker_safe(ticker)
    tname = par84.TICKER_NAMES[ticker]
    for key, cfg in FREQ_CONFIGS.items():
        df = fetch(ticker, cfg['period'], cfg['interval'])
        if df is None:
            continue
        fname = f"4-{key}-{safe}.html"
        plot_rsi(df, 'RSI', fname, f'{ticker} · {tname} · {cfg["label"]}')

        # Backtest
        rsi = compute_rsi(df['Close'])
        rsi_prev = rsi.shift(1)
        buy_sig  = (rsi_prev < 30) & (rsi >= 30)
        sell_sig = (rsi_prev > 70) & (rsi <= 70)
        bt_df, bt_stats = run_backtest(df, buy_sig, sell_sig)
        bt_fname = f"4-bt-{key}-{safe}.html"
        plot_backtest(bt_df, bt_stats, 'RSI', bt_fname, f'{ticker} · {tname} · {cfg["label"]}')

print("\n✓ Script 4 done")
