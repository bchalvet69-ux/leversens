import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import par84

images = par84.img_path + '6-'

tickers = par84.TICKERS
period = '1y'

try:
    df = yf.download(tickers, period=period, auto_adjust=True)['Close'].dropna()
    print(f"Downloaded {len(df)} days of data")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Calculate daily returns
returns = df.pct_change().dropna()

print(f"Returns shape: {returns.shape}")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")

# ===== GRAPH 6-1: Correlation Heatmap =====
corr_matrix = returns.corr()

print(f"\nCorrelation Matrix:")
print(corr_matrix)

labels = [par84.TICKER_NAMES[t] for t in corr_matrix.columns]

fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, '#ef4444'],    # -1 red (bad: strong negative corr)
            [0.15, '#f87171'],
            [0.35, '#fbbf24'],   # transition
            [0.5, '#22c55e'],    # 0 green (ideal: uncorrelated)
            [0.65, '#fbbf24'],   # transition
            [0.85, '#f87171'],
            [1.0, '#ef4444'],    # +1 red (bad: strong positive corr)
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=13, color='#f1f5f9'),
        colorbar=dict(
            title=dict(text='ρ', font=dict(size=14, color='#94a3b8')),
            tickfont=dict(size=10, color='#94a3b8'),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            thickness=12,
            len=0.6,
        ),
        hovertemplate='%{x} vs %{y}<br>ρ = %{z:.3f}<extra></extra>',
    ),
    layout=par84.layout,
)

fig.update_layout(
    title=par84.dash_title('CORRELATION MATRIX', '5 Assets · Daily Returns · 1 Year'),
    xaxis=dict(
        tickfont=dict(size=11, color='#cbd5e1'),
        side='bottom',
        tickangle=0,
    ),
    yaxis=dict(
        tickfont=dict(size=11, color='#cbd5e1'),
        autorange='reversed',
    ),
    margin=dict(l=100, r=40, t=60, b=60),
)
fig.write_html(images + '1.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 6-1.html: Correlation Heatmap")

print("\n✓ Script 6.py completed successfully!")
