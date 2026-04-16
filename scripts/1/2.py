import pandas as pd
import numpy as np
import yfinance as yf
import par84
import plotly.graph_objects as go
from _helpers import get_rangebreaks

# ============================================================================
# Script 2 - Ichimoku Cloud
# Dashboard (2-1.html): SPY 6-month Ichimoku
# Analytics: one chart per asset × 2 frequencies (intraday 60d/1h + monthly 1y/1d)
#            → files 2-intraday-SPY.html, 2-monthly-SPY.html, etc.
# ============================================================================


def compute_ichimoku(df):
    """Return a DataFrame with Ichimoku components aligned on index."""
    out = df.copy()
    high_9  = df['High'].rolling(9).max()
    low_9   = df['Low'].rolling(9).min()
    out['tenkan'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(26).max()
    low_26  = df['Low'].rolling(26).min()
    out['kijun'] = (high_26 + low_26) / 2

    out['span_a_raw'] = (out['tenkan'] + out['kijun']) / 2
    high_52 = df['High'].rolling(52).max()
    low_52  = df['Low'].rolling(52).min()
    out['span_b_raw'] = (high_52 + low_52) / 2

    out['chikou'] = df['Close'].shift(-26)
    return out


def plot_ichimoku(df, title, fname, freq_label=""):
    ichi = compute_ichimoku(df)

    # Build a future index of 26 bars using the median delta of the existing index
    if len(df.index) > 1:
        delta = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    else:
        delta = pd.Timedelta(days=1)
    future_index = pd.DatetimeIndex(
        [df.index[-1] + (i + 1) * delta for i in range(26)]
    )
    extended = df.index.append(future_index)

    span_a = pd.Series(index=extended, dtype=float)
    span_b = pd.Series(index=extended, dtype=float)
    span_a.iloc[26:26+len(df)] = ichi['span_a_raw'].values
    span_b.iloc[26:26+len(df)] = ichi['span_b_raw'].values

    # Split bullish/bearish cloud
    bull_a = span_a.where(span_a >= span_b)
    bull_b = span_b.where(span_a >= span_b)
    bear_a = span_a.where(span_a < span_b)
    bear_b = span_b.where(span_a < span_b)

    fig = go.Figure(layout=par84.layout)

    # Bullish cloud (green tint)
    fig.add_scatter(x=extended, y=bull_a, mode='lines',
                    line=dict(width=0), showlegend=False, name='')
    fig.add_scatter(x=extended, y=bull_b, mode='lines',
                    line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(34, 197, 94, 0.18)', name='Bullish cloud')

    # Bearish cloud (red tint)
    fig.add_scatter(x=extended, y=bear_a, mode='lines',
                    line=dict(width=0), showlegend=False, name='')
    fig.add_scatter(x=extended, y=bear_b, mode='lines',
                    line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(239, 68, 68, 0.18)', name='Bearish cloud')

    # Candlesticks
    fig.add_candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=par84.BULL), fillcolor=par84.BULL),
        decreasing=dict(line=dict(color=par84.BEAR), fillcolor=par84.BEAR),
        name='Price', showlegend=False,
    )

    fig.add_scatter(x=df.index, y=ichi['tenkan'], mode='lines',
                    line=dict(color='#f59e0b', width=1.4), name='Tenkan (9)')
    fig.add_scatter(x=df.index, y=ichi['kijun'], mode='lines',
                    line=dict(color='#a78bfa', width=1.4), name='Kijun (26)')
    fig.add_scatter(x=df.index, y=ichi['chikou'], mode='lines',
                    line=dict(color='#fbbf24', width=1, dash='dot'), name='Chikou (-26)')

    rb = get_rangebreaks(df)
    fig.update_layout(
        paper_bgcolor=par84.CHART_BG,
        plot_bgcolor=par84.CHART_BG,
        title=par84.dash_title('ICHIMOKU', f'{title} · {freq_label}' if freq_label else title),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor='rgba(255,255,255,0.06)',
                   showspikes=True, spikemode='across', spikedash='dot',
                   spikecolor='#3b82f6', rangebreaks=rb),
        yaxis=dict(title='Price', gridcolor='rgba(255,255,255,0.06)'),
        legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
    )
    fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ {fname}")


def fetch(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 10:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Europe/Paris').tz_localize(None)
        return df
    except Exception as e:
        print(f"    ✗ {ticker} {period}/{interval}: {e}")
        return None


# ═══ Dashboard: SPY 6-month ═══
print("▶ Dashboard (2-1.html): SPY 6-month Ichimoku")
df = fetch('SPY', '6mo', '1d')
if df is not None:
    plot_ichimoku(df, 'SPY · S&P 500', '2-1.html', 'Daily · 6m')


# ═══ Analytics: for each asset × 2 frequencies ═══
FREQ_CONFIGS = {
    'intraday': {'period': '60d', 'interval': '1h', 'label': 'Intraday · 60d × 1h'},
    'monthly':  {'period': '1y',  'interval': '1d', 'label': 'Monthly view · 1y × 1d'},
}

print("\n▶ Analytics: per asset × 2 frequencies")
for ticker in par84.TICKERS:
    safe = ticker.replace('=', '').replace('-', '')
    tname = par84.TICKER_NAMES[ticker]
    for key, cfg in FREQ_CONFIGS.items():
        df = fetch(ticker, cfg['period'], cfg['interval'])
        if df is None:
            continue
        fname = f'2-{key}-{safe}.html'
        plot_ichimoku(df, f'{ticker} · {tname}', fname, cfg['label'])

print("\n✓ Script 2 done")
