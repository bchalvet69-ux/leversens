"""Shared helpers for Dashboard 1 scripts — TradingView-like blue theme."""
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import par84


def fetch(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 15:
            return None
        # Convert to Paris time so SPY shows 15h30-22h00, not 9h30-16h00
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Europe/Paris').tz_localize(None)
        return df
    except Exception as e:
        print(f"    ✗ {ticker} {period}/{interval}: {e}")
        return None


def get_rangebreaks(df):
    """Auto-detect rangebreaks from the data to eliminate gaps (TradingView-like).
    - Daily: skip weekends
    - Intraday: skip weekends + non-trading hours (auto-detected)
    """
    breaks = [dict(bounds=["sat", "mon"])]  # always skip weekends

    if len(df) < 2:
        return breaks

    # Detect if intraday (multiple candles per day)
    median_diff = pd.Series(df.index).diff().dropna().median()
    is_intraday = median_diff < pd.Timedelta(hours=12)

    if is_intraday:
        # Check if 24/7 (like BTC) or has specific trading hours
        unique_hours = df.index.hour.nunique()
        if unique_hours < 20:  # Not 24/7 trading
            times = df.index.hour + df.index.minute / 60.0
            open_time = float(times.min())              # e.g. 15.5 for 15:30
            close_time = float(np.ceil(times.max()))   # e.g. 22.0
            if close_time != open_time:
                breaks.append(dict(bounds=[close_time, open_time], pattern="hour"))

    return breaks


FREQ_CONFIGS = {
    'intraday': {'period': '60d', 'interval': '1h', 'label': 'Intraday · 60d × 1h'},
    'monthly':  {'period': '1y',  'interval': '1d', 'label': 'Daily · 1y'},
}


def ticker_safe(t):
    return t.replace('=', '').replace('-', '')


# ═══════════════════════ BACKTEST ENGINE ═══════════════════════

def run_backtest(df, buy_signal, sell_signal, initial_capital=10000):
    """
    Generic backtest: long-only strategy vs buy-and-hold.
    Returns df with 'strategy_value' and 'buy_hold_value' columns,
    plus a stats dict.
    """
    capital = initial_capital
    position = 0       # 0 = cash, 1 = in position
    shares = 0.0
    equity = []
    trades = 0

    for i in range(len(df)):
        price = df['Close'].iloc[i]

        if buy_signal.iloc[i] and position == 0:
            position = 1
            shares = capital / price
            capital = 0.0
            trades += 1

        elif sell_signal.iloc[i] and position == 1:
            position = 0
            capital = shares * price
            shares = 0.0

        portfolio_value = capital if position == 0 else shares * price
        equity.append(portfolio_value)

    df = df.copy()
    df['strategy_value'] = equity
    bh_shares = initial_capital / df['Close'].iloc[0]
    df['buy_hold_value'] = bh_shares * df['Close']

    strat_ret = (df['strategy_value'].iloc[-1] / initial_capital - 1) * 100
    bh_ret    = (df['buy_hold_value'].iloc[-1] / initial_capital - 1) * 100

    stats = dict(
        initial=initial_capital,
        final_strategy=df['strategy_value'].iloc[-1],
        final_bh=df['buy_hold_value'].iloc[-1],
        strategy_return=strat_ret,
        bh_return=bh_ret,
        trades=trades,
        outperforms=strat_ret > bh_ret,
    )
    return df, stats


def plot_backtest(df, stats, strategy_name, fname, subtitle=""):
    """
    Plot equity curves: Strategy vs Buy & Hold.
    Styled to match the site theme.
    """
    fig = go.Figure()

    # Strategy curve
    fig.add_scatter(
        x=df.index, y=df['strategy_value'],
        mode='lines', name=f'{strategy_name} Strategy',
        line=dict(color='#f59e0b', width=2.2),
        hovertemplate='%{y:,.0f}<extra>Strategy</extra>',
    )

    # Buy & Hold curve
    fig.add_scatter(
        x=df.index, y=df['buy_hold_value'],
        mode='lines', name='Buy & Hold',
        line=dict(color='#60a5fa', width=2, dash='dot'),
        hovertemplate='%{y:,.0f}<extra>Buy & Hold</extra>',
    )

    # Fill between to highlight outperformance
    fig.add_scatter(
        x=df.index, y=df['strategy_value'],
        mode='lines', line=dict(width=0), showlegend=False,
        hoverinfo='skip',
    )
    fig.add_scatter(
        x=df.index, y=df['buy_hold_value'],
        mode='lines', line=dict(width=0), showlegend=False,
        fill='tonexty',
        fillcolor='rgba(245,158,11,0.08)',
        hoverinfo='skip',
    )

    # Starting capital reference line
    fig.add_hline(
        y=stats['initial'], line=dict(color='rgba(148,163,184,0.3)', width=1, dash='dot'),
        annotation_text=f"Initial: {stats['initial']:,.0f}",
        annotation_position='bottom left',
        annotation_font=dict(size=10, color='#64748b'),
    )

    # Stats annotation box
    color = '#22c55e' if stats['outperforms'] else '#ef4444'
    winner = strategy_name if stats['outperforms'] else 'Buy & Hold'
    ann_text = (
        f"<b>{strategy_name}:</b> {stats['strategy_return']:+.2f}% "
        f"({stats['final_strategy']:,.0f})<br>"
        f"<b>Buy & Hold:</b> {stats['bh_return']:+.2f}% "
        f"({stats['final_bh']:,.0f})<br>"
        f"<b>Trades:</b> {stats['trades']}  ·  "
        f"<b style='color:{color}'>Winner: {winner}</b>"
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=ann_text, showarrow=False,
        font=dict(size=11, family='monospace', color='#cbd5e1'),
        align='left', bgcolor='rgba(15,23,41,0.85)',
        bordercolor='rgba(59,130,246,0.3)', borderwidth=1, borderpad=8,
    )

    rb = get_rangebreaks(df)
    layout = base_layout(f'{strategy_name} BACKTEST', subtitle, rangebreaks=rb)
    layout['yaxis']['tickprefix'] = ''
    layout['yaxis']['tickformat'] = ',.0f'
    fig.update_layout(**layout)

    fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
    winner_tag = '✓' if stats['outperforms'] else '✗'
    print(f"  ✓ {fname}  ({strategy_name} {stats['strategy_return']:+.1f}% vs B&H {stats['bh_return']:+.1f}% {winner_tag})")


def base_layout(title, subtitle="", rangebreaks=None):
    layout = dict(
        template='plotly_dark',
        paper_bgcolor=par84.CHART_BG,
        plot_bgcolor=par84.CHART_BG,
        title=par84.dash_title(title, subtitle),
        margin=dict(l=10, r=60, t=60, b=30),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showspikes=True, spikemode='across',
                   spikecolor='#3b82f6', spikedash='dot', spikethickness=1,
                   rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', side='right',
                   showspikes=True, spikemode='across', spikecolor='#3b82f6',
                   spikedash='dot', spikethickness=1, tickfont=dict(color='#cbd5e1', size=10)),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        font=dict(family='ui-monospace, Menlo, Consolas, monospace', color='#cbd5e1'),
        colorway=par84.COLORS,
    )
    if rangebreaks:
        layout['xaxis']['rangebreaks'] = rangebreaks
    return layout
