import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import par84

images = par84.img_path + '5-'

# CAPM analysis: 5 assets vs SPY benchmark
tickers = par84.TICKERS
benchmark = 'SPY'
other_assets = [t for t in tickers if t != benchmark]
period = '2y'

# Risk-free rate (annualized, ~4% per year)
rf_annual = 0.04
trading_days = 252
rolling_window = 60

try:
    df = yf.download(tickers, period=period, auto_adjust=True)['Close'].dropna()
    print(f"Downloaded {len(df)} days of data")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Calculate daily returns
daily_returns = df.pct_change().dropna()

# Benchmark returns (SPY)
benchmark_returns = daily_returns[benchmark]

# ===== GRAPH 5-1: Rolling Beta (60 days) for Each Asset vs SPY =====
fig = go.Figure(layout=par84.layout)

for i, ticker in enumerate(other_assets):
    try:
        # Calculate rolling beta
        asset_returns = daily_returns[ticker]
        rolling_cov = asset_returns.rolling(rolling_window).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(rolling_window).var()
        rolling_beta = rolling_cov / rolling_var

        fig.add_scatter(
            x=rolling_beta.index,
            y=rolling_beta.values,
            name=par84.TICKER_NAMES[ticker],
            line=dict(color=par84.COLORS[i+1]),  # Skip SPY color
            mode='lines'
        )
        print(f"{ticker}: Mean Rolling Beta = {rolling_beta.mean():.4f}")
    except Exception as e:
        print(f"Error calculating rolling beta for {ticker}: {e}")

fig.update_layout(
    title=par84.dash_title('ROLLING BETA', f'vs SPY · {rolling_window}-day window'),
    xaxis_title="Date",
    yaxis_title="Beta",
    hovermode='x unified'
)
fig.write_html(images + '1.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 5-1.html: Rolling Beta")

# ===== Calculate Static Beta and CAPM Returns =====
results = []

for ticker in other_assets:
    try:
        asset_returns = daily_returns[ticker]

        # Static beta
        cov = asset_returns.cov(benchmark_returns)
        var = benchmark_returns.var()
        beta = cov / var

        # Annual returns
        asset_annual = (1 + asset_returns.mean()) ** trading_days - 1
        benchmark_annual = (1 + benchmark_returns.mean()) ** trading_days - 1

        # CAPM expected return: rf + beta * (rm - rf)
        capm_return = rf_annual + beta * (benchmark_annual - rf_annual)

        # Sharpe ratio
        sharpe = (asset_annual - rf_annual) / asset_returns.std() / np.sqrt(trading_days)

        results.append({
            'Ticker': ticker,
            'Name': par84.TICKER_NAMES[ticker],
            'Beta': beta,
            'Annual Return': asset_annual,
            'CAPM Expected Return': capm_return,
            'Sharpe Ratio': sharpe
        })

        print(f"\n{ticker}:")
        print(f"  Beta: {beta:.4f}")
        print(f"  Annual Return: {asset_annual:.4f}")
        print(f"  CAPM Expected Return: {capm_return:.4f}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
    except Exception as e:
        print(f"Error for {ticker}: {e}")

# ===== GRAPH 5-2: Scatter Plot Alpha vs Beta =====
alpha_list = []
beta_list = []
names_list = []

for ticker in other_assets:
    try:
        asset_returns = daily_returns[ticker]

        # Static beta
        cov = asset_returns.cov(benchmark_returns)
        var = benchmark_returns.var()
        beta = cov / var

        # Annual returns
        asset_annual = (1 + asset_returns.mean()) ** trading_days - 1
        benchmark_annual = (1 + benchmark_returns.mean()) ** trading_days - 1

        # CAPM expected return
        capm_return = rf_annual + beta * (benchmark_annual - rf_annual)

        # Alpha = actual return - CAPM expected return
        alpha = asset_annual - capm_return

        alpha_list.append(alpha)
        beta_list.append(beta)
        names_list.append(par84.TICKER_NAMES[ticker])
    except Exception as e:
        print(f"Error calculating alpha for {ticker}: {e}")

fig = go.Figure(layout=par84.layout)

fig.add_scatter(
    x=beta_list,
    y=alpha_list,
    mode='markers+text',
    marker=dict(size=12, color='#f59e0b'),
    text=names_list,
    textposition='top center',
    name='Assets'
)

# Add security market line (SML)
beta_range = np.array([min(beta_list) - 0.5, max(beta_list) + 0.5])
benchmark_annual = (1 + benchmark_returns.mean()) ** trading_days - 1
sml = rf_annual + beta_range * (benchmark_annual - rf_annual)
fig.add_scatter(
    x=beta_range,
    y=sml,
    mode='lines',
    name='Security Market Line',
    line=dict(color='red', dash='dash')
)

# Add zero line
fig.add_hline(y=0, line_dash="dot", line_color="gray")

fig.update_layout(
    title=par84.dash_title('ALPHA vs BETA', 'CAPM Analysis · 4 Assets vs SPY'),
    xaxis_title="Beta",
    yaxis_title="Alpha (Excess Return)",
    hovermode='closest'
)
fig.write_html(images + '2.html', config=par84.config, include_plotlyjs="cdn")
print(f"✓ Saved 5-2.html: Alpha vs Beta Scatter")

# ===== GRAPH 5-3: Summary Table (HTML) =====
results_df = pd.DataFrame(results)

table_html = f"""
<html>
<head>
<style>
    body {{
        color: #cbd5e1;
        background-color: #111827;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        padding: 20px;
    }}
    h2 {{ color: #f59e0b; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        max-width: 900px;
    }}
    th, td {{
        border: 1px solid rgba(255,255,255,0.1);
        padding: 12px;
        text-align: right;
    }}
    th {{
        background-color: rgba(167,139,250,0.25);
        color: #e2e8f0;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: rgba(255,255,255,0.03);
    }}
    tr:nth-child(odd) {{
        background-color: rgba(255,255,255,0.06);
    }}
    td:first-child, th:first-child {{
        text-align: left;
    }}
</style>
</head>
<body>
<h2>CAPM Analysis Summary</h2>
<table>
    <tr>
        <th>Ticker</th>
        <th>Name</th>
        <th>Beta</th>
        <th>Annual Return</th>
        <th>CAPM Expected Return</th>
        <th>Sharpe Ratio</th>
    </tr>
"""

for _, row in results_df.iterrows():
    table_html += f"""
    <tr>
        <td>{row['Ticker']}</td>
        <td>{row['Name']}</td>
        <td>{row['Beta']:.4f}</td>
        <td>{row['Annual Return']:.4f}</td>
        <td>{row['CAPM Expected Return']:.4f}</td>
        <td>{row['Sharpe Ratio']:.4f}</td>
    </tr>
"""

table_html += """
</table>
</body>
</html>
"""

with open(images + '3.html', 'w', encoding='utf-8') as f:
    f.write(table_html)

print(f"✓ Saved 5-3.html: CAPM Summary Table")

print("\n✓ Script 5.py completed successfully!")

