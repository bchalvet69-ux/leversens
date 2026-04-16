import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
import par84

# ============================================================================
# Script 2 — Monte Carlo Simulation (GBM)
# Per-asset: 2-dash-{SAFE}.html  (1000 simulations, 30d horizon)
# Dashboard default: 2-dash-SPY.html
# ============================================================================

SAFE_MAP = {'SPY': 'SPY', 'BTC-USD': 'BTCUSD', 'CL=F': 'CLF', 'GOLD': 'GOLD', 'NEM': 'NEM'}

t_intervals = 30   # 30 days forecast
simulations = 1000
period = '1y'

print(f"▶ Monte Carlo · {simulations} GBM paths · {t_intervals}d horizon")

for ticker in par84.TICKERS:
    safe = SAFE_MAP[ticker]
    tname = par84.TICKER_NAMES[ticker]
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)['Close']
        if len(df) < 30:
            print(f"  ✗ {ticker}: not enough data")
            continue

        returns = np.log(df / df.shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()

        # GBM simulation
        daily_log_ret = mu + sigma * norm.ppf(np.random.rand(t_intervals, simulations))
        daily_simple_ret = np.exp(daily_log_ret)

        price_paths = np.zeros((t_intervals, simulations))
        price_paths[0] = df.iloc[-1]
        for t in range(1, t_intervals):
            price_paths[t] = price_paths[t - 1] * daily_simple_ret[t]

        # Stats
        final_prices = price_paths[-1]
        median_path = np.median(price_paths, axis=1)
        p5 = np.percentile(price_paths, 5, axis=1)
        p25 = np.percentile(price_paths, 25, axis=1)
        p75 = np.percentile(price_paths, 75, axis=1)
        p95 = np.percentile(price_paths, 95, axis=1)

        # --- Plot ---
        fig = go.Figure(layout=par84.layout)

        # Sample 200 paths for display (1000 is too heavy to render)
        display_idx = np.random.choice(simulations, size=min(200, simulations), replace=False)
        for i in display_idx:
            fig.add_scatter(
                x=np.arange(t_intervals), y=price_paths[:, i],
                mode='lines',
                line=dict(color='rgba(96,165,250,0.06)', width=0.8),
                showlegend=False, hoverinfo='skip',
            )

        # Confidence bands (shaded)
        x_days = list(range(t_intervals))
        x_rev = list(reversed(x_days))

        # 90% band
        fig.add_scatter(
            x=x_days + x_rev,
            y=list(p95) + list(reversed(p5)),
            fill='toself', fillcolor='rgba(59,130,246,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True, name='90% CI',
            hoverinfo='skip',
        )

        # 50% band
        fig.add_scatter(
            x=x_days + x_rev,
            y=list(p75) + list(reversed(p25)),
            fill='toself', fillcolor='rgba(59,130,246,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True, name='50% CI',
            hoverinfo='skip',
        )

        # Median path
        fig.add_scatter(
            x=x_days, y=median_path,
            mode='lines', name='Median',
            line=dict(color='#f1f5f9', width=2),
        )

        # 5th / 95th edges
        fig.add_scatter(
            x=x_days, y=p5, mode='lines', name='P5',
            line=dict(color='#ef4444', width=1, dash='dot'),
        )
        fig.add_scatter(
            x=x_days, y=p95, mode='lines', name='P95',
            line=dict(color='#22c55e', width=1, dash='dot'),
        )

        var5_final = np.percentile(final_prices, 5)
        pct_change = (np.median(final_prices) / df.iloc[-1] - 1) * 100

        fig.update_layout(
            title=par84.dash_title('MONTE CARLO',
                                   f'{ticker} · {tname} · {simulations} sims · {t_intervals}d'),
            xaxis_title='Days',
            yaxis_title='Price ($)',
            hovermode='x unified',
            margin=dict(l=10, r=50, t=60, b=30),
            yaxis=dict(side='right', gridcolor='rgba(255,255,255,0.06)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        )

        fname = f'2-dash-{safe}.html'
        fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
        print(f"  ✓ {fname}  (median Δ={pct_change:+.1f}%, VaR5=${var5_final:.2f})")

    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

print(f"\n✓ Script 2.py completed successfully!")
