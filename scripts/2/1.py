import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import par84

# ============================================================================
# Script 1 — GARCH Conditional Volatility + Forecast
# Per-asset: 1-dash-{SAFE}.html  (conditional vol + 30d forecast)
# Dashboard default: 1-dash-SPY.html
# ============================================================================

SAFE_MAP = {'SPY': 'SPY', 'BTC-USD': 'BTCUSD', 'CL=F': 'CLF', 'GOLD': 'GOLD', 'NEM': 'NEM'}

period = '2y'
forecast_horizon = 30

print("▶ GARCH · Per-asset conditional volatility + forecast")

for ticker in par84.TICKERS:
    safe = SAFE_MAP[ticker]
    tname = par84.TICKER_NAMES[ticker]
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)['Close'].dropna()
        if len(df) < 100:
            print(f"  ✗ {ticker}: not enough data ({len(df)} rows)")
            continue

        log_ret = np.log(df / df.shift(1)).dropna()
        scaled = log_ret * 100  # scale for arch lib

        # Fit GARCH(1,1)
        model = arch_model(scaled, vol='Garch', p=1, q=1, mean='Constant')
        res = model.fit(disp='off')

        # Historical conditional volatility (annualized)
        cond_vol = res.conditional_volatility / 100 * np.sqrt(252)

        # Forecast
        fcast = res.forecast(horizon=forecast_horizon)
        fcast_var = fcast.variance.iloc[-1].values  # variance per day (in scaled units)
        fcast_vol = np.sqrt(fcast_var) / 100 * np.sqrt(252)  # annualized

        # Build forecast dates
        last_date = cond_vol.index[-1]
        fcast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

        # --- Plot ---
        fig = go.Figure(layout=par84.layout)

        # Historical conditional vol
        fig.add_scatter(
            x=cond_vol.index, y=cond_vol.values,
            mode='lines', name='Conditional Vol',
            line=dict(color='#60a5fa', width=1.5),
        )

        # Forecast
        fig.add_scatter(
            x=fcast_dates, y=fcast_vol,
            mode='lines', name=f'Forecast ({forecast_horizon}d)',
            line=dict(color='#f59e0b', width=2, dash='dash'),
        )

        # Vertical separator
        fig.add_vline(
            x=last_date.timestamp() * 1000,
            line_dash='dot', line_color='rgba(148,163,184,0.5)', line_width=1,
        )

        mean_vol = cond_vol.mean()
        fig.update_layout(
            title=par84.dash_title('GARCH VOLATILITY',
                                   f'{ticker} · {tname} · Annualized · {forecast_horizon}d Forecast'),
            yaxis_title='Annualized Volatility',
            hovermode='x unified',
            margin=dict(l=10, r=50, t=60, b=30),
            yaxis=dict(tickformat='.0%', side='right',
                       gridcolor='rgba(255,255,255,0.06)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        )

        fname = f'1-dash-{safe}.html'
        fig.write_html(par84.img_path + fname, config=par84.config, include_plotlyjs="cdn")
        print(f"  ✓ {fname}  (mean vol = {mean_vol:.1%})")

    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

print("\n✓ Script 1.py completed successfully!")
