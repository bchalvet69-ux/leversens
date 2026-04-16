import plotly.graph_objects as go

img_path = "../../img/1/"

TICKERS = ['SPY', 'BTC-USD', 'CL=F', 'GOLD', 'NEM']

# Multi-hue palette for line series (avoids all-blue)
COLORS  = ['#f59e0b', '#a78bfa', '#34d399', '#f472b6', '#60a5fa']

TICKER_NAMES = {
    'SPY':     'S&P 500 ETF',
    'BTC-USD': 'Bitcoin',
    'CL=F':    'Crude Oil',
    'GOLD':    'Barrick Gold',
    'NEM':     'Newmont Mining',
}

# Bull / bear candle colors
BULL = '#22c55e'   # green
BEAR = '#ef4444'   # red
ACCENT = '#3b82f6'
ACCENT_SOFT = '#60a5fa'

# Slightly lighter background so charts are readable on dashboard
CHART_BG = '#111827'

layout = go.Layout(
    template="plotly_dark",
    title=dict(x=0.5, xanchor='center', yanchor='top'),
    colorway=COLORS,
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=52, pad=0),
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    font=dict(color='#cbd5e1'),
)


def dash_title(technique, subtitle=""):
    """Centered HTML title: technique name only."""
    t = f"<b style='font-size:14px;letter-spacing:1.5px'>{technique}</b>"
    return dict(text=t, x=0.5, xanchor='center', y=0.95)

config = {
    'displayModeBar': False,
    'scrollZoom': True,
}
