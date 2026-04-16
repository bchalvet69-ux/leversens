"""Shared helpers for Dashboard 1 scripts — TradingView-like blue theme."""
import numpy as np
import pandas as pd
import yfinance as yf
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
