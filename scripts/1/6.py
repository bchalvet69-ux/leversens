import numpy as np
import pandas as pd
import yfinance as yf
import par84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _helpers import get_rangebreaks

# ============================================================================
# Script 6 - ICT Breaker Block + OHLC Statistical Mapping (NEM)
# Intraday only: 5min and 15min candles, ~3 trading days
# OHLC levels drawn per session (each day has its own Open + distribution/
# manipulation projections)
# ============================================================================

SYMBOL = 'NEM'
IMAGES = par84.img_path


# ═══════════════════════ DATA FETCH ═══════════════════════
def fetch(symbol, period, interval):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Europe/Paris').tz_localize(None)
    return df.dropna()


def trim_to_n_days(df, n=5):
    """Keep only the last n trading days of intraday data."""
    dates = df.index.normalize().unique()
    if len(dates) <= n:
        return df
    cutoff = dates[-n]
    return df[df.index >= cutoff]


# ═══════════════════════ INDICATORS ═══════════════════════
def compute_atr(df, period=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_ict_structures(df, atr_mult=1.0, max_zone_bars=15):
    """
    Return list of dicts describing Order Blocks and Breaker Blocks.
     - Bullish OB = last bearish candle before a strong bullish push
     - Bearish OB = last bullish candle before a strong bearish push
     - Breaker   = OB whose opposing boundary has been violated
    Zone rectangles are capped at max_zone_bars from their start so they
    don't stretch across the entire chart.
    """
    df = df.copy()
    df['atr'] = compute_atr(df)
    df['body'] = (df['Close'] - df['Open']).abs()
    df['dir']  = np.where(df['Close'] >= df['Open'], 1, -1)

    zones = []
    for i in range(2, len(df) - 2):
        row  = df.iloc[i]
        prev = df.iloc[i-1]
        atr  = row['atr']
        if pd.isna(atr) or atr == 0:
            continue

        # Cap end index: start + max_zone_bars, clamped to last bar
        end_i = min(i - 1 + max_zone_bars, len(df) - 1)

        if prev['dir'] == -1 and row['dir'] == 1 and row['body'] >= atr * atr_mult:
            zones.append({
                'type': 'bullish_ob',
                'start': df.index[i-1], 'end': df.index[end_i],
                'top': prev['High'], 'bot': prev['Low'], 'i0': i-1,
            })
        if prev['dir'] == 1 and row['dir'] == -1 and row['body'] >= atr * atr_mult:
            zones.append({
                'type': 'bearish_ob',
                'start': df.index[i-1], 'end': df.index[end_i],
                'top': prev['High'], 'bot': prev['Low'], 'i0': i-1,
            })

    for z in zones:
        later = df.iloc[z['i0']+2:]
        if later.empty:
            z['broken'] = False
            continue
        if z['type'] == 'bullish_ob':
            breaks = later[later['Close'] < z['bot']]
            if not breaks.empty:
                z['type'] = 'bearish_breaker'
                z['break_time'] = breaks.index[0]
                z['broken'] = True
                # Breaker ends shortly after the break point
                break_pos = df.index.get_loc(z['break_time'])
                z['end'] = df.index[min(break_pos + 8, len(df) - 1)]
            else:
                z['broken'] = False
        else:
            breaks = later[later['Close'] > z['top']]
            if not breaks.empty:
                z['type'] = 'bullish_breaker'
                z['break_time'] = breaks.index[0]
                z['broken'] = True
                break_pos = df.index.get_loc(z['break_time'])
                z['end'] = df.index[min(break_pos + 8, len(df) - 1)]
            else:
                z['broken'] = False

    zones = sorted(zones, key=lambda z: z['start'])[-8:]
    return zones


def compute_ohlc_stats(symbol, lookback=20):
    """Compute avg distribution & manipulation from DAILY data (last N days)."""
    daily = yf.Ticker(symbol).history(period='3mo', interval='1d', auto_adjust=True)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_convert('Europe/Paris').tz_localize(None)
    daily = daily.dropna()
    if len(daily) < lookback + 1:
        return None
    recent = daily.iloc[-lookback-1:-1]
    is_bull = recent['Close'] >= recent['Open']
    distrib = np.where(is_bull, recent['High'] - recent['Open'], recent['Open'] - recent['Low'])
    manip   = np.where(is_bull, recent['Open'] - recent['Low'], recent['High'] - recent['Open'])
    return dict(avg_d=float(np.nanmean(distrib)), avg_m=float(np.nanmean(manip)))


# ═══════════════════════ STYLING ═══════════════════════
BULL = par84.BULL
BEAR = par84.BEAR

def tv_layout(title, subtitle=""):
    return dict(
        template='plotly_dark',
        paper_bgcolor=par84.CHART_BG,
        plot_bgcolor=par84.CHART_BG,
        title=par84.dash_title(title, subtitle),
        margin=dict(l=10, r=55, t=70, b=40),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            showspikes=True, spikemode='across', spikecolor='#3b82f6',
            spikedash='dot', spikethickness=1,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            side='right',
            showspikes=True, spikemode='across', spikecolor='#3b82f6',
            spikedash='dot', spikethickness=1,
            tickfont=dict(color='#cbd5e1', size=10),
        ),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        font=dict(family='ui-monospace, Menlo, Consolas, monospace', color='#cbd5e1'),
    )


def add_candles(fig, df, row=1, col=1):
    fig.add_candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=BULL, width=1), fillcolor=BULL),
        decreasing=dict(line=dict(color=BEAR, width=1), fillcolor=BEAR),
        showlegend=False, name='Price',
        row=row, col=col,
    )


def add_volume(fig, df, row, col=1):
    colors = np.where(df['Close'] >= df['Open'], 'rgba(34,197,94,0.45)', 'rgba(239,68,68,0.40)')
    fig.add_bar(
        x=df.index, y=df['Volume'],
        marker=dict(color=colors, line=dict(width=0)),
        showlegend=False, name='Volume',
        row=row, col=col,
    )


ZONE_STYLE = {
    'bullish_ob':       dict(fill='rgba(34,197,94,0.12)',   edge='rgba(34,197,94,0.75)',  label='Bullish OB'),
    'bearish_ob':       dict(fill='rgba(239,68,68,0.12)',   edge='rgba(239,68,68,0.75)',  label='Bearish OB'),
    'bullish_breaker':  dict(fill='rgba(52,211,153,0.22)',  edge='rgba(52,211,153,0.90)', label='Bullish Breaker'),
    'bearish_breaker':  dict(fill='rgba(244,114,182,0.22)', edge='rgba(244,114,182,0.90)', label='Bearish Breaker'),
}


def add_ict_zones(fig, df, zones, row=1, col=1):
    seen_labels = set()
    for z in zones:
        style = ZONE_STYLE[z['type']]
        show_legend = z['type'] not in seen_labels
        seen_labels.add(z['type'])

        # Zone rectangle (compact)
        fig.add_shape(
            type='rect',
            xref=f"x{'' if row==1 else row}", yref=f"y{'' if row==1 else row}",
            x0=z['start'], x1=z['end'],
            y0=z['bot'], y1=z['top'],
            fillcolor=style['fill'],
            line=dict(color=style['edge'], width=1.4,
                      dash='solid' if 'breaker' in z['type'] else 'dash'),
            layer='below',
            row=row, col=col,
        )
        # Invisible scatter for legend entry only
        fig.add_scatter(
            x=[z['start']], y=[z['top']],
            mode='markers', marker=dict(size=0.1, color=style['edge']),
            name=style['label'], showlegend=show_legend,
            legendgroup=z['type'],
            row=row, col=col,
        )
        # BOS triangle at break point (breakers only)
        if z.get('break_time') is not None and 'breaker' in z['type']:
            marker_y = z['top'] if z['type'] == 'bullish_breaker' else z['bot']
            fig.add_scatter(
                x=[z['break_time']], y=[marker_y],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if z['type'] == 'bullish_breaker' else 'triangle-down',
                    size=10, color=style['edge'], line=dict(color='#fff', width=1),
                ),
                showlegend=False, hovertext='BOS',
                row=row, col=col,
            )


def add_ohlc_levels_per_session(fig, df, levels, row=1, col=1):
    """Draw OHLC statistical levels for each trading session (day).
    Each session gets its own Open + ±Distribution + ±Manipulation lines."""
    if not levels:
        return
    avg_d = levels['avg_d']
    avg_m = levels['avg_m']

    dates = df.index.normalize().unique()
    session_colors = ['#e2e8f0', '#93c5fd', '#c4b5fd']  # white, blue, purple for each session

    legend_added = False
    for idx, date in enumerate(dates):
        session = df[df.index.normalize() == date]
        if session.empty:
            continue
        open_val = session.iloc[0]['Open']
        x0 = session.index[0]
        x1 = session.index[-1]
        base_color = session_colors[idx % len(session_colors)]

        lines_def = [
            (open_val + avg_d, 'D+', '#f59e0b', 'dot'),
            (open_val + avg_m, 'M+', '#a78bfa', 'dash'),
            (open_val,         'O',  base_color, 'solid'),
            (open_val - avg_m, 'M-', '#a78bfa', 'dash'),
            (open_val - avg_d, 'D-', '#f59e0b', 'dot'),
        ]
        is_last = (date == dates[-1])
        for y, label, color, style in lines_def:
            fig.add_shape(
                type='line',
                xref=f"x{'' if row==1 else row}", yref=f"y{'' if row==1 else row}",
                x0=x0, x1=x1, y0=y, y1=y,
                line=dict(color=color, width=1.2, dash=style),
                layer='above',
                row=row, col=col,
            )

        # Labels pinned to the end of each line on the last session only
        if is_last:
            label_x = [x1] * len(lines_def)
            label_y = [lv[0] for lv in lines_def]
            label_t = [f"  {lv[1]}" for lv in lines_def]
            label_c = [lv[2] for lv in lines_def]
            fig.add_scatter(
                x=label_x, y=label_y,
                mode='text', text=label_t,
                textposition='middle right',
                textfont=dict(size=10, family='monospace, sans-serif',
                              color=label_c),
                showlegend=False, hoverinfo='skip',
                row=row, col=col,
            )

        # Session separator — vertical dotted line at session start
        if idx > 0:
            fig.add_vline(
                x=x0,
                line=dict(color='rgba(148,163,184,0.3)', width=1, dash='dot'),
                row=row, col=col,
            )


# ═══════════════════════ CHART BUILDERS ═══════════════════════
def build_ict_chart(df, out, title, subtitle):
    zones = find_ict_structures(df)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )
    add_candles(fig, df, row=1)
    add_ict_zones(fig, df, zones, row=1)
    if 'Volume' in df.columns:
        add_volume(fig, df, row=2)

    rb = get_rangebreaks(df)
    layout = tv_layout(title, subtitle)
    layout['xaxis']['rangebreaks'] = rb
    fig.update_layout(**layout)
    fig.update_yaxes(title_text='', row=1, col=1, side='right', gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text='Vol', row=2, col=1, side='right', gridcolor='rgba(255,255,255,0.04)',
                     tickfont=dict(size=9, color='#666'))
    fig.update_xaxes(rangeslider=dict(visible=False))

    fig.write_html(IMAGES + out, config=par84.config, include_plotlyjs="cdn")
    n_bb = sum(1 for z in zones if 'breaker' in z['type'])
    print(f"  ✓ {out}  ({len(zones)} zones · {n_bb} breakers)")


def build_ohlc_chart(df, ohlc_stats, out, title, subtitle):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )
    add_candles(fig, df, row=1)
    add_ohlc_levels_per_session(fig, df, ohlc_stats, row=1)
    if 'Volume' in df.columns:
        add_volume(fig, df, row=2)

    rb = get_rangebreaks(df)
    layout = tv_layout(title, subtitle)
    layout['xaxis']['rangebreaks'] = rb
    fig.update_layout(**layout)
    fig.update_yaxes(title_text='', row=1, col=1, side='right', gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text='Vol', row=2, col=1, side='right', gridcolor='rgba(255,255,255,0.04)',
                     tickfont=dict(size=9, color='#666'))
    fig.update_xaxes(rangeslider=dict(visible=False))

    fig.write_html(IMAGES + out, config=par84.config, include_plotlyjs="cdn")
    if ohlc_stats:
        print(f"  ✓ {out}  (avg D={ohlc_stats['avg_d']:.2f}, avg M={ohlc_stats['avg_m']:.2f})")
    else:
        print(f"  ✓ {out}  (no OHLC stats)")


def build_combined_chart(df, ohlc_stats, out, title, subtitle):
    zones = find_ict_structures(df)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.02,
    )
    add_candles(fig, df, row=1)
    add_ict_zones(fig, df, zones, row=1)
    add_ohlc_levels_per_session(fig, df, ohlc_stats, row=1)
    if 'Volume' in df.columns:
        add_volume(fig, df, row=2)

    rb = get_rangebreaks(df)
    layout = tv_layout(title, subtitle)
    layout['xaxis']['rangebreaks'] = rb
    fig.update_layout(**layout)
    fig.update_yaxes(title_text='', row=1, col=1, side='right', gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text='Vol', row=2, col=1, side='right', gridcolor='rgba(255,255,255,0.04)',
                     tickfont=dict(size=9, color='#666'))
    fig.update_xaxes(rangeslider=dict(visible=False))

    fig.write_html(IMAGES + out, config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ {out}  (combined view)")


# ═══════════════════════ RUN ═══════════════════════
SAFE_MAP = {'SPY': 'SPY', 'BTC-USD': 'BTCUSD', 'CL=F': 'CLF', 'GOLD': 'GOLD', 'NEM': 'NEM'}

INTRADAY_CONFIGS = {
    '5m':  {'period': '10d',  'interval': '5m',  'label': '5 days × 5 min'},
    '15m': {'period': '10d',  'interval': '15m', 'label': '5 days × 15 min'},
}

# ── Per-asset × per-timeframe × per-view (analytics page) ──
print("▶ ICT + OHLC · All assets × timeframes")
for ticker in par84.TICKERS:
    safe = SAFE_MAP[ticker]
    tname = par84.TICKER_NAMES[ticker]
    asset_stats = compute_ohlc_stats(ticker)
    if asset_stats:
        print(f"\n  {ticker} OHLC stats: avg_d={asset_stats['avg_d']:.2f}, avg_m={asset_stats['avg_m']:.2f}")

    for key, cfg in INTRADAY_CONFIGS.items():
        df = fetch(ticker, cfg['period'], cfg['interval'])
        if df is None or df.empty:
            print(f"  ✗ {ticker} {key} data unavailable")
            continue
        df = trim_to_n_days(df, n=5)

        # ICT only
        build_ict_chart(df, f'6-ict-{key}-{safe}.html', 'ICT BREAKER BLOCKS',
                        f'{ticker} · {tname} · {cfg["label"]}')
        # OHLC only
        build_ohlc_chart(df, asset_stats, f'6-ohlc-{key}-{safe}.html', 'OHLC MAPPING',
                         f'{ticker} · {tname} · {cfg["label"]}')
        # Combined
        build_combined_chart(df, asset_stats, f'6-{key}-{safe}.html', 'ICT + OHLC',
                             f'{ticker} · {tname} · {cfg["label"]}')

    # Dashboard variant (15m combined)
    df_dash = fetch(ticker, '10d', '15m')
    if df_dash is not None and not df_dash.empty:
        df_dash = trim_to_n_days(df_dash, n=3)
        build_combined_chart(df_dash, asset_stats, f'6-dash-{safe}.html', 'ICT + OHLC',
                             f'{ticker} · {tname} · 5d × 15min')

# ── Default dashboard chart (6-1.html) = NEM 15m combined ──
df_default = fetch(SYMBOL, '10d', '15m')
if df_default is not None:
    df_default = trim_to_n_days(df_default, n=3)
    default_stats = compute_ohlc_stats(SYMBOL)
    build_combined_chart(df_default, default_stats, '6-1.html', 'ICT + OHLC',
                         f'{SYMBOL} · Newmont · 5d × 15min')

print("\n✓ Script 6 done")
