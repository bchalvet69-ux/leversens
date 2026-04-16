import pandas as pd
import numpy as np
import yfinance as yf
import par84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ============================================================================
# Script 1 - Base 100 Performance (5 assets, 3 ans) + Daily Returns Analytics
#            + Per-asset daily returns + Yahoo Finance News
# ============================================================================

# Download 3y daily closes for each ticker and keep each series on its own calendar
data_dict = {}
for ticker in par84.TICKERS:
    try:
        df = yf.Ticker(ticker).history(period='3y', interval='1d', auto_adjust=True)
        s = df['Close'].dropna()
        # Drop timezone to make indexes comparable across tickers
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        data_dict[ticker] = s
        print(f"✓ {ticker}: {len(s)} rows")
    except Exception as e:
        print(f"✗ {ticker}: {e}")

# ───── DASHBOARD CHART (1-1.html): Base 100 on common trading-day axis ─────
df_combined = pd.DataFrame(data_dict)
df_combined = df_combined.ffill().bfill().dropna(how='all')
base100 = df_combined.divide(df_combined.iloc[0]).multiply(100)

fig1 = go.Figure(layout=par84.layout)
for i, ticker in enumerate(par84.TICKERS):
    if ticker in base100.columns:
        fig1.add_scatter(
            x=base100.index,
            y=base100[ticker],
            mode='lines',
            name=par84.TICKER_NAMES[ticker],
            line=dict(color=par84.COLORS[i], width=2),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        )

fig1.update_layout(
    title=par84.dash_title('GLOBAL OVERVIEW', '5 Assets · 3 Years'),
    xaxis=dict(title='', gridcolor='rgba(255,255,255,0.05)', showspikes=True, spikemode='across',
               spikedash='dot', spikecolor='rgba(140,70,255,0.5)'),
    yaxis=dict(title='Base 100', gridcolor='rgba(255,255,255,0.05)'),
    legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5),
    hovermode='x unified',
)
fig1.write_html(par84.img_path + '1-1.html', config=par84.config, include_plotlyjs="cdn")
print("✓ 1-1.html saved")

# ───── ANALYTICS CHART (1-2.html): Daily Returns — all 5 assets stacked ─────
fig2 = make_subplots(
    rows=5, cols=1, shared_xaxes=False,
    subplot_titles=[par84.TICKER_NAMES[t] for t in par84.TICKERS],
    vertical_spacing=0.05,
)

for i, ticker in enumerate(par84.TICKERS):
    s = data_dict.get(ticker)
    if s is None or s.empty:
        continue
    returns = s.pct_change().dropna() * 100
    if returns.empty:
        continue
    colors = np.where(returns.values >= 0, '#22c55e', '#ef4444')
    fig2.add_trace(
        go.Bar(
            x=returns.index,
            y=returns.values,
            marker=dict(color=colors, line=dict(width=0)),
            showlegend=False,
            name=par84.TICKER_NAMES[ticker],
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ),
        row=i+1, col=1,
    )
    fig2.update_yaxes(title_text='%', row=i+1, col=1, gridcolor='rgba(255,255,255,0.05)',
                      zeroline=True, zerolinecolor='rgba(255,255,255,0.25)')
    fig2.update_xaxes(row=i+1, col=1, gridcolor='rgba(255,255,255,0.03)')

fig2.update_layout(
    template='plotly_dark',
    paper_bgcolor=par84.CHART_BG,
    plot_bgcolor=par84.CHART_BG,
    title=par84.dash_title('DAILY RETURNS', '5 Assets · 3 Years'),
    height=1100,
    margin=dict(l=40, r=20, t=60, b=40),
    bargap=0,
)
for ann in fig2['layout']['annotations']:
    ann['font'] = dict(size=12, color='#60a5fa')

fig2.write_html(par84.img_path + '1-2.html', config=par84.config, include_plotlyjs="cdn")
print("✓ 1-2.html saved")

# ───── PER-ASSET DAILY RETURNS (1-ret-{SAFE}.html) ─────
SAFE = {'SPY': 'SPY', 'BTC-USD': 'BTCUSD', 'CL=F': 'CLF', 'GOLD': 'GOLD', 'NEM': 'NEM'}

for ticker in par84.TICKERS:
    s = data_dict.get(ticker)
    if s is None or s.empty:
        continue
    returns = s.pct_change().dropna() * 100
    if returns.empty:
        continue
    colors_arr = np.where(returns.values >= 0, '#22c55e', '#ef4444')

    fig = go.Figure(layout=par84.layout)
    fig.add_trace(go.Bar(
        x=returns.index,
        y=returns.values,
        marker=dict(color=colors_arr, line=dict(width=0)),
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=par84.dash_title('DAILY RETURNS'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='%', gridcolor='rgba(255,255,255,0.05)',
                   zeroline=True, zerolinecolor='rgba(255,255,255,0.25)'),
        margin=dict(l=40, r=20, t=60, b=40),
        bargap=0,
    )
    safe = SAFE[ticker]
    fig.write_html(par84.img_path + f'1-ret-{safe}.html', config=par84.config, include_plotlyjs="cdn")
    print(f"  ✓ 1-ret-{safe}.html saved")

print("✓ Per-asset daily returns done")

# ───── YAHOO FINANCE NEWS (news.json) — via yf.Search() ─────
from datetime import datetime

news_data = {}
for ticker in par84.TICKERS:
    safe = SAFE[ticker]
    try:
        results = yf.Search(ticker, news_count=10)
        items = results.news[:10] if results.news else []
        articles = []
        for n in items:
            title = n.get('title', '')
            link = n.get('link', '')
            publisher = n.get('publisher', '')
            # Date: providerPublishTime is a unix timestamp
            ts = n.get('providerPublishTime', n.get('publish_time', 0))
            date_str = ''
            thumbnail = ''
            try:
                date_str = datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                date_str = str(ts) if ts else ''
            # Thumbnail
            thumb = n.get('thumbnail', {})
            if isinstance(thumb, dict):
                resolutions = thumb.get('resolutions', [])
                if resolutions and isinstance(resolutions[0], dict):
                    thumbnail = resolutions[0].get('url', '')
            if title and link:
                articles.append({
                    'title': title,
                    'link': link,
                    'publisher': publisher,
                    'date': date_str,
                    'thumbnail': thumbnail,
                })
        # Sort by date, most recent first
        articles.sort(key=lambda a: a.get('date', ''), reverse=True)
        news_data[safe] = articles
        print(f"  ✓ {ticker}: {len(articles)} news articles")
    except Exception as e:
        news_data[safe] = []
        print(f"  ✗ {ticker} news: {e}")

with open(par84.img_path + 'news.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=2)
print("✓ news.json saved")

# Also generate a JS file for file:// protocol compatibility (fetch() blocked on local files)
with open(par84.img_path + 'news-data.js', 'w', encoding='utf-8') as f:
    f.write('var NEWS_DATA = ')
    json.dump(news_data, f, ensure_ascii=False, indent=2)
    f.write(';\n')
print("✓ news-data.js saved")
