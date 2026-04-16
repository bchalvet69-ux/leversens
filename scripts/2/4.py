import numpy as np
import pandas as pd
import json
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from pathlib import Path
import par84

images = par84.img_path + '4-'

# Pair trading: GOLD vs NEM
tickers = ['GOLD', 'NEM']
y_ticker, x_ticker = tickers
period = '1y'
interval = '1d'

try:
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=True)['Close'].dropna()
    print(f"Downloaded {len(df)} days of data for {tickers}")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# OLS regression: GOLD = alpha + beta * NEM + epsilon
X = sm.add_constant(df[[x_ticker]], has_constant='add')
y = df[y_ticker]

model = sm.OLS(y, X).fit()
print(model.summary())

# ── Model validation metrics ──
r_squared = model.rsquared
f_stat = model.fvalue
f_pval = model.f_pvalue
durbin_watson = float(sm.stats.durbin_watson(model.resid))

# R² > 0.7 and F-test significant → model is valid
r2_ok = r_squared > 0.7
f_ok = f_pval < 0.05
dw_ok = 1.5 < durbin_watson < 2.5  # DW near 2 = no autocorrelation

model_valid = r2_ok and f_ok

# Build validation info for JS
pair_validation = {
    'r_squared': round(float(r_squared), 4),
    'f_stat': round(float(f_stat), 2),
    'f_pval': float(f_pval),
    'durbin_watson': round(float(durbin_watson), 4),
    'r2_ok': bool(r2_ok),
    'f_ok': bool(f_ok),
    'dw_ok': bool(dw_ok),
    'valid': bool(model_valid),
}

# Get residuals and test for stationarity
resid = model.resid.dropna()
adf_stat, pval, *_ = adfuller(resid)

print(f"\nADF Test on Residuals:")
print(f"  Statistic: {adf_stat:.4f}")
print(f"  p-value: {pval:.6f}")
if pval < 0.05:
    print(f"  -> Residuals are stationary (good for pair trading)")
else:
    print(f"  -> Residuals are NOT stationary (pair may not be cointegrated)")

pair_validation['adf_pval'] = round(float(pval), 6)
pair_validation['cointegrated'] = bool(pval < 0.05)

# ===== GRAPH 4-1: OLS Summary as HTML with model validation badge =====
valid_color = '#22c55e' if model_valid else '#ef4444'
valid_icon = '✓' if model_valid else '⚠'
valid_text = 'Model Valid' if model_valid else 'Model Questionable'

r2_verdict = "validates the model (must be &gt; 0.70)" if r2_ok else "does not validate the model (must be &gt; 0.70)"
f_verdict = "validates the model (must be &gt; ~4.0)" if f_ok else "does not validate the model (must be &gt; ~4.0)"
p_ok = f_ok  # same check — p-value of the F-test
p_verdict = "validates the model (must be &lt; 0.05)" if p_ok else "does not validate the model (must be &lt; 0.05)"

# Extract coefficient data
coef_const = model.params['const']
coef_nem = model.params[x_ticker]
se_const = model.bse['const']
se_nem = model.bse[x_ticker]
t_const = model.tvalues['const']
t_nem = model.tvalues[x_ticker]
p_const = model.pvalues['const']
p_nem = model.pvalues[x_ticker]
ci_const = model.conf_int().loc['const']
ci_nem = model.conf_int().loc[x_ticker]
n_obs = int(model.nobs)

summary_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    color:#cbd5e1; font-family: ui-monospace, Menlo, Consolas, monospace;
    background-color: #111827; padding:24px;
  }}
  h2 {{
    font-size:16px; letter-spacing:1.5px; color:#f1f5f9;
    margin-bottom:16px; font-weight:700;
  }}
  .badge {{
    padding:16px 20px; border-radius:10px; font-size:13px; line-height:2;
    margin-bottom:20px;
    background: {'rgba(34,197,94,0.08)' if model_valid else 'rgba(239,68,68,0.08)'};
    border: 1px solid {'rgba(34,197,94,0.25)' if model_valid else 'rgba(239,68,68,0.25)'};
  }}
  .badge .line {{ display:block; }}
  .badge .val {{ font-weight:700; color:#f1f5f9; font-size:14px; }}
  .badge .ok {{ color:#86efac; }}
  .badge .ko {{ color:#fca5a5; }}
  .badge .explain {{
    display:block; margin-top:10px; padding-top:10px;
    border-top:1px solid rgba(255,255,255,0.08);
    font-size:12px; color:#94a3b8; line-height:1.6;
    font-style:italic;
  }}
  table {{
    border-collapse:collapse; width:100%; max-width:800px;
    margin-top:20px;
  }}
  th, td {{
    padding:10px 14px; text-align:right;
    border-bottom:1px solid rgba(255,255,255,0.06);
  }}
  th {{
    background:rgba(59,130,246,0.12); color:#93c5fd;
    font-size:11px; letter-spacing:1px; text-transform:uppercase;
    font-weight:700;
  }}
  td {{ font-size:13px; }}
  td:first-child, th:first-child {{ text-align:left; }}
  tr:hover td {{ background:rgba(59,130,246,0.05); }}
  .section-label {{
    font-size:11px; letter-spacing:2px; text-transform:uppercase;
    color:#475569; margin:24px 0 10px; font-weight:700;
  }}
  .meta {{ font-size:12px; color:#64748b; margin-top:8px; }}
</style>
</head>
<body>

<h2>OLS REGRESSION &mdash; {y_ticker} vs {x_ticker}</h2>

<div class="badge">
  <span class="line">
    <span class="val">R&sup2; = {r_squared:.4f}</span>
    &rarr; <span class="{'ok' if r2_ok else 'ko'}">{r2_verdict}</span>
  </span>
  <span class="line">
    <span class="val">F-statistic = {float(f_stat):.1f}</span>
    &rarr; <span class="{'ok' if f_ok else 'ko'}">{f_verdict}</span>
  </span>
  <span class="line">
    <span class="val">P-value = {float(f_pval):.2e}</span>
    &rarr; <span class="{'ok' if p_ok else 'ko'}">{p_verdict}</span>
  </span>
  <span class="explain">
    R&sup2; measures how much of {y_ticker}'s variance is explained by {x_ticker},
    the F-statistic tests whether the linear relationship is statistically significant,
    and the P-value confirms that this result is not due to chance.
  </span>
</div>

<div class="section-label">Coefficients</div>
<table>
  <tr>
    <th>Variable</th><th>Coefficient</th><th>Std Error</th><th>t-stat</th><th>P-value</th><th>95% CI</th>
  </tr>
  <tr>
    <td>Intercept</td>
    <td>{coef_const:.4f}</td>
    <td>{se_const:.4f}</td>
    <td>{t_const:.3f}</td>
    <td>{p_const:.4f}</td>
    <td>[{ci_const.iloc[0]:.3f}, {ci_const.iloc[1]:.3f}]</td>
  </tr>
  <tr>
    <td>{x_ticker}</td>
    <td>{coef_nem:.4f}</td>
    <td>{se_nem:.4f}</td>
    <td>{t_nem:.3f}</td>
    <td>{p_nem:.4f}</td>
    <td>[{ci_nem.iloc[0]:.3f}, {ci_nem.iloc[1]:.3f}]</td>
  </tr>
</table>

<div class="section-label">Model Info</div>
<table>
  <tr><th style="width:50%">Metric</th><th>Value</th></tr>
  <tr><td>Observations</td><td>{n_obs}</td></tr>
  <tr><td>R-squared</td><td>{r_squared:.4f}</td></tr>
  <tr><td>Adj. R-squared</td><td>{model.rsquared_adj:.4f}</td></tr>
  <tr><td>F-statistic</td><td>{float(f_stat):.1f}</td></tr>
  <tr><td>Prob (F-statistic)</td><td>{float(f_pval):.2e}</td></tr>
  <tr><td>Log-Likelihood</td><td>{model.llf:.2f}</td></tr>
  <tr><td>AIC</td><td>{model.aic:.1f}</td></tr>
  <tr><td>BIC</td><td>{model.bic:.1f}</td></tr>
  <tr><td>Durbin-Watson</td><td>{durbin_watson:.4f}</td></tr>
</table>

<div class="meta">Model: OLS &middot; Method: Least Squares &middot; Dep. Variable: {y_ticker}</div>

</body>
</html>"""

Path(images + '1.html').write_text(summary_html, encoding="utf-8")
print(f"✓ Saved 4-1.html: OLS Summary (styled)")

# Write pair validation JS
js_content = f"var PAIR_VALIDATION = {json.dumps(pair_validation, indent=2)};"
with open(par84.img_path + 'pair-validation.js', 'w') as f:
    f.write(js_content)
print(f"✓ Saved pair-validation.js")

# Calculate predicted values and residuals
predicted = model.predict(X)
resid_series = y - predicted

# Calculate Z-score
mu = resid.mean()
std = resid.std(ddof=1)
z_score = (resid_series - mu) / std

# Create signals dataframe — TIGHTER bands at ±1.5σ
Z_THRESH = 1.5

signals = df.copy()
signals['predicted'] = predicted
signals['residual'] = resid_series
signals['z_score'] = z_score
signals['upper_band'] = Z_THRESH
signals['lower_band'] = -Z_THRESH

# Generate trading signals (crossing threshold)
z_prev = signals['z_score'].shift(1)
z_curr = signals['z_score']

short_entry = (z_prev <= Z_THRESH) & (z_curr > Z_THRESH)
long_entry = (z_prev >= -Z_THRESH) & (z_curr < -Z_THRESH)

signals['signal_y'] = np.nan
signals.loc[long_entry, 'signal_y'] = 1
signals.loc[short_entry, 'signal_y'] = -1
signals['signal_y'] = signals['signal_y'].ffill().fillna(0)

# Generate position changes
signals['position_y'] = signals['signal_y'].diff().fillna(0)
signals['signal_x'] = -signals['signal_y']
signals['position_x'] = signals['signal_x'].diff().fillna(0)

# ===== GRAPH 4-2: Z-score with Bands + Buy/Sell Signals =====
fig = go.Figure(layout=par84.layout)

fig.add_scatter(
    x=signals.index,
    y=signals['z_score'],
    name='Z-score',
    line=dict(color='#f59e0b', width=2),
    mode='lines'
)

# Upper band
fig.add_scatter(
    x=signals.index,
    y=signals['upper_band'],
    name=f'+{Z_THRESH}σ (Short)',
    line=dict(color='#ef4444', width=1, dash='dash'),
    mode='lines'
)

# Lower band with fill
fig.add_scatter(
    x=signals.index,
    y=signals['lower_band'],
    name=f'-{Z_THRESH}σ (Long)',
    line=dict(color='#22c55e', width=1, dash='dash'),
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(245, 158, 11, 0.06)'
)

# Zero line
fig.add_hline(y=0, line_dash="dot", line_color="gray")

# Buy signals on z-score chart
long_signals = signals[signals['position_y'] > 0]
if not long_signals.empty:
    fig.add_scatter(
        x=long_signals.index,
        y=signals.loc[long_signals.index, 'z_score'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', color='#22c55e', size=14,
                    line=dict(width=1.5, color='#16a34a')),
    )

# Sell signals on z-score chart
short_signals = signals[signals['position_y'] < 0]
if not short_signals.empty:
    fig.add_scatter(
        x=short_signals.index,
        y=signals.loc[short_signals.index, 'z_score'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='#ef4444', size=14,
                    line=dict(width=1.5, color='#dc2626')),
    )

fig.update_layout(
    title=par84.dash_title('PAIR TRADING · Z-SCORE', f'{y_ticker} vs {x_ticker} · ±{Z_THRESH}σ'),
    xaxis_title="Date",
    yaxis_title="Z-Score",
    hovermode='x unified'
)
fig.write_html(images + '2.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 4-2.html: Z-Score Bands + Signals")

# ===== GRAPH 4-3: Normalized Prices (Base 100) with Signals =====
df_normalized = df / df.iloc[0] * 100

fig = go.Figure(layout=par84.layout)

# Add both assets
fig.add_scatter(
    x=df_normalized.index,
    y=df_normalized[y_ticker],
    name=f'{y_ticker} (base 100)',
    line=dict(color='#a78bfa', width=2),
    mode='lines'
)

fig.add_scatter(
    x=df_normalized.index,
    y=df_normalized[x_ticker],
    name=f'{x_ticker} (base 100)',
    line=dict(color='#f59e0b', width=2, dash='dot'),
    mode='lines'
)

# Add long signals (triangles up)
if not long_signals.empty:
    fig.add_scatter(
        x=long_signals.index,
        y=df_normalized.loc[long_signals.index, y_ticker],
        mode='markers',
        name=f'Long {y_ticker}',
        marker=dict(symbol='triangle-up', color='limegreen', size=12)
    )

# Add short signals (triangles down)
if not short_signals.empty:
    fig.add_scatter(
        x=short_signals.index,
        y=df_normalized.loc[short_signals.index, y_ticker],
        mode='markers',
        name=f'Short {y_ticker}',
        marker=dict(symbol='triangle-down', color='red', size=12)
    )

fig.update_layout(
    title=par84.dash_title('PAIR TRADING · SIGNALS', f'{y_ticker} vs {x_ticker} · Base 100'),
    xaxis_title="Date",
    yaxis_title="Normalized Price (Base 100)",
    hovermode='x unified'
)
fig.write_html(images + '3.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 4-3.html: Normalized Prices with Signals")

# ===== GRAPH 4-4: P&L Cumulative from Pair Trading Strategy =====
# Calculate daily returns
returns_y = df[y_ticker].pct_change().fillna(0)
returns_x = df[x_ticker].pct_change().fillna(0)

# P&L: long y & short x when signal_y = 1, short y & long x when signal_y = -1
strategy_returns = signals['signal_y'].shift(1) * returns_y + (-signals['signal_y'].shift(1)) * returns_x
cumulative_pnl = (1 + strategy_returns).cumprod() - 1

fig = go.Figure(layout=par84.layout)

fig.add_scatter(
    x=cumulative_pnl.index,
    y=cumulative_pnl.values * 100,
    name='Cumulative P&L (%)',
    line=dict(color='#34d399', width=2),
    mode='lines',
    fill='tozeroy',
    fillcolor='rgba(52, 211, 153, 0.12)'
)

# Add zero line
fig.add_hline(y=0, line_dash="dash", line_color="gray")

fig.update_layout(
    title=par84.dash_title('PAIR TRADING · P&L', f'{y_ticker} vs {x_ticker} · Cumulative'),
    xaxis_title="Date",
    yaxis_title="Cumulative Return (%)",
    hovermode='x unified'
)
fig.write_html(images + '4.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 4-4.html: Cumulative P&L")

print(f"\nPair Trading Summary:")
print(f"  Total Return: {cumulative_pnl.iloc[-1]*100:.2f}%")
print(f"  Number of signals: {len(long_signals) + len(short_signals)}")
print(f"  R²: {r_squared:.4f}, F-stat: {float(f_stat):.1f}, Model valid: {model_valid}")

print("\n✓ Script 4.py completed successfully!")
