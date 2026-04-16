import numpy as np
import pandas as pd
import json
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
import par84

# ============================================================================
# Script 3 — ARIMA Forecast + Residuals + ACF (all per-asset)
# Per-asset:
#   3-dash-{SAFE}.html   (forecast)
#   3-resid-{SAFE}.html  (residuals)
#   3-acf-{SAFE}.html    (ACF)
# Model validation JS: arima-validation.js
# ============================================================================

SAFE_MAP = {'SPY': 'SPY', 'BTC-USD': 'BTCUSD', 'CL=F': 'CLF', 'GOLD': 'GOLD', 'NEM': 'NEM'}

period = '2y'
forecast_days = 30

validation_data = {}

print("▶ ARIMA · Per-asset forecast, residuals, ACF")

for ticker in par84.TICKERS:
    safe = SAFE_MAP[ticker]
    tname = par84.TICKER_NAMES[ticker]
    try:
        tk_df = yf.Ticker(ticker).history(period=period, auto_adjust=True)['Close']
        if len(tk_df) < 60:
            print(f"  ✗ {ticker}: not enough data ({len(tk_df)} rows)")
            continue

        # ADF test
        adf = adfuller(tk_df, autolag='AIC')
        adf_stat = float(adf[0])
        adf_pval = float(adf[1])
        d = 1 if adf_pval > 0.05 else 0

        # Fit ARIMA(2,d,2)
        mdl = ARIMA(tk_df, order=(2, d, 2)).fit()
        residuals = mdl.resid

        # ── Multi-criteria validation (strict) ──

        # 1. Ljung-Box at multiple lags (residuals = white noise?)
        lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        lb_pval_10 = float(lb['lb_pvalue'].iloc[0])
        lb_pval_20 = float(lb['lb_pvalue'].iloc[1])
        ljung_ok = lb_pval_10 > 0.05 and lb_pval_20 > 0.05

        # 2. ARCH test (heteroscedasticity in residuals — volatility clustering)
        try:
            arch_test = het_arch(residuals.dropna(), nlags=10)
            arch_pval = float(arch_test[1])  # LM test p-value
        except Exception:
            arch_pval = 0.0  # fail safe
        arch_ok = arch_pval > 0.05  # no ARCH effects

        # 3. Normality (Jarque-Bera)
        jb_stat, jb_pval = jarque_bera(residuals.dropna())[:2]
        jb_pval = float(jb_pval)
        normality_ok = jb_pval > 0.05

        # 4. Coefficient significance (all AR/MA terms p < 0.05)
        pvalues = mdl.pvalues
        # exclude 'sigma2' (variance) from significance check
        coef_pvals = pvalues.drop('sigma2', errors='ignore')
        all_signif = bool((coef_pvals < 0.05).all()) if len(coef_pvals) > 0 else False

        # 5. ADF on residuals (must be stationary)
        adf_resid = adfuller(residuals.dropna(), autolag='AIC')
        adf_resid_pval = float(adf_resid[1])
        resid_stationary = adf_resid_pval < 0.05

        # AIC / BIC
        aic = float(mdl.aic)
        bic = float(mdl.bic)

        # ── Verdict: must pass ALL checks to be valid ──
        checks = {
            'ljung_box': ljung_ok,
            'no_arch': arch_ok,
            'normality': normality_ok,
            'coef_signif': all_signif,
            'resid_stationary': resid_stationary,
        }
        n_passed = sum(checks.values())
        n_total = len(checks)
        is_valid = n_passed == n_total

        # Build failure reasons
        failures = []
        if not ljung_ok:
            failures.append('residual autocorrelation (Ljung-Box)')
        if not arch_ok:
            failures.append('ARCH effects / volatility clustering')
        if not normality_ok:
            failures.append('non-normal residuals (Jarque-Bera)')
        if not all_signif:
            failures.append('insignificant coefficients')
        if not resid_stationary:
            failures.append('non-stationary residuals')

        if is_valid:
            message = f'All {n_total} validation tests passed — forecast is reliable.'
        else:
            message = f'Model fails {n_total - n_passed}/{n_total} tests: {", ".join(failures)}. Forecast should be used with extreme caution.'

        validation_data[safe] = {
            'ticker': ticker,
            'name': tname,
            'order': f'({2},{d},{2})',
            'adf_stat': round(adf_stat, 4),
            'adf_pval': round(adf_pval, 4),
            'stationary': adf_pval <= 0.05,
            'd': d,
            'ljung_box_10': round(lb_pval_10, 4),
            'ljung_box_20': round(lb_pval_20, 4),
            'arch_pval': round(arch_pval, 4),
            'jb_pval': round(jb_pval, 4),
            'coef_signif': all_signif,
            'resid_stationary': resid_stationary,
            'checks_passed': n_passed,
            'checks_total': n_total,
            'aic': round(aic, 1),
            'bic': round(bic, 1),
            'valid': is_valid,
            'message': message,
        }

        # ═══ 1. Forecast chart ═══
        fc = mdl.get_forecast(steps=forecast_days)
        fc_df = fc.conf_int(alpha=0.05)
        fc_df['forecast'] = fc.predicted_mean
        fc_dates = pd.date_range(start=tk_df.index[-1], periods=forecast_days + 1, freq='D')[1:]
        fc_df.index = fc_dates
        fc_df.columns = ['lower', 'upper', 'forecast']

        fig = go.Figure(layout=par84.layout)
        fig.add_scatter(x=tk_df.index, y=tk_df.values, name='Historical',
                        line=dict(color='#60a5fa', width=1.5), mode='lines')
        fig.add_scatter(x=fc_df.index, y=fc_df['forecast'], name='Forecast',
                        line=dict(color='#f59e0b', width=2, dash='dash'), mode='lines')
        fig.add_scatter(x=fc_df.index, y=fc_df['upper'], mode='lines',
                        line=dict(color='rgba(0,0,0,0)'), showlegend=False)
        fig.add_scatter(x=fc_df.index, y=fc_df['lower'], mode='lines',
                        line=dict(color='rgba(0,0,0,0)'), fill='tonexty',
                        fillcolor='rgba(167,139,250,0.15)', name='95% CI')
        fig.update_layout(
            title=par84.dash_title('ARIMA FORECAST',
                                   f'{ticker} · {tname} · ARIMA{validation_data[safe]["order"]} · {forecast_days}d'),
            yaxis_title='Price ($)', hovermode='x unified',
            margin=dict(l=10, r=50, t=60, b=30),
            yaxis=dict(side='right', gridcolor='rgba(255,255,255,0.06)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        )
        fig.write_html(par84.img_path + f'3-dash-{safe}.html', config=par84.config, include_plotlyjs="cdn")
        print(f"  ✓ 3-dash-{safe}.html")

        # ═══ 2. Residuals chart ═══
        fig = go.Figure(layout=par84.layout)
        fig.add_scatter(
            x=residuals.index, y=residuals.values,
            mode='markers', name='Residuals',
            marker=dict(color='#a78bfa', size=4, opacity=0.6),
        )
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(148,163,184,0.4)')
        fig.update_layout(
            title=par84.dash_title('ARIMA RESIDUALS', f'{ticker} · {tname}'),
            yaxis_title='Residual', hovermode='closest',
            margin=dict(l=10, r=50, t=60, b=30),
            yaxis=dict(side='right', gridcolor='rgba(255,255,255,0.06)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        )
        fig.write_html(par84.img_path + f'3-resid-{safe}.html', config=par84.config, include_plotlyjs="cdn")
        print(f"  ✓ 3-resid-{safe}.html")

        # ═══ 3. ACF chart ═══
        lags = 30
        acf_values = acf(residuals, nlags=lags, fft=False)
        ci = 1.96 / np.sqrt(len(residuals))

        fig = go.Figure(layout=par84.layout)
        colors = ['#ef4444' if abs(v) > ci else '#34d399' for v in acf_values]
        fig.add_bar(
            x=np.arange(len(acf_values)), y=acf_values,
            name='ACF', marker=dict(color=colors),
        )
        fig.add_hline(y=ci, line_dash='dash', line_color='rgba(239,68,68,0.6)')
        fig.add_hline(y=-ci, line_dash='dash', line_color='rgba(239,68,68,0.6)')
        fig.update_layout(
            title=par84.dash_title('ACF RESIDUALS', f'{ticker} · {tname} · Autocorrelation'),
            xaxis_title='Lag', yaxis_title='Autocorrelation', hovermode='closest',
            margin=dict(l=10, r=50, t=60, b=30),
            yaxis=dict(side='right', gridcolor='rgba(255,255,255,0.06)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        )
        fig.write_html(par84.img_path + f'3-acf-{safe}.html', config=par84.config, include_plotlyjs="cdn")
        print(f"  ✓ 3-acf-{safe}.html")

    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

# Write validation data as JS for the analytics page
js_content = f"var ARIMA_VALIDATION = {json.dumps(validation_data, indent=2)};"
with open(par84.img_path + 'arima-validation.js', 'w') as f:
    f.write(js_content)
print(f"  ✓ arima-validation.js")

print("\n✓ Script 3.py completed successfully!")
