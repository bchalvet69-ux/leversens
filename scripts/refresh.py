#!/usr/bin/env python3
"""
Master refresh script — runs all chart-generating scripts in order.
Called by GitHub Actions or manually: python scripts/refresh.py
"""
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

SCRIPTS = [
    # Dashboard 1
    ("scripts/1", "1.py",  "Global Overview (Base 100 + News)"),
    ("scripts/1", "2.py",  "Ichimoku"),
    ("scripts/1", "3.py",  "MACD"),
    ("scripts/1", "4.py",  "RSI"),
    ("scripts/1", "5.py",  "Bollinger Bands"),
    ("scripts/1", "6.py",  "ICT & OHLC Mapping"),
    # Dashboard 2
    ("scripts/2", "6.py",  "Correlation Matrix"),
    ("scripts/2", "1.py",  "GARCH Volatility"),
    ("scripts/2", "2.py",  "Monte Carlo"),
    ("scripts/2", "3.py",  "ARIMA Forecast"),
    ("scripts/2", "4.py",  "Pair Trading"),
    ("scripts/2", "5.py",  "CAPM / Rolling Beta"),
    ("scripts/2", "7_btc_onchain.py", "BTC On-Chain"),
]

def run_script(cwd, script, label):
    full_cwd = ROOT / cwd
    print(f"\n{'='*60}")
    print(f"  {label}  ({cwd}/{script})")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=full_cwd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"  *** FAILED (exit code {result.returncode}) ***")
        return False
    return True

def main():
    print("LEVERSENS — Full Data Refresh")
    print(f"Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    ok = 0
    fail = 0
    for cwd, script, label in SCRIPTS:
        if run_script(cwd, script, label):
            ok += 1
        else:
            fail += 1

    # Write last-updated timestamp
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    ts_file = ROOT / "img" / "last-updated.js"
    ts_file.write_text(f'var LAST_UPDATED = "{ts}";')
    print(f"\n✓ Wrote {ts_file}")

    print(f"\n{'='*60}")
    print(f"  DONE — {ok} succeeded, {fail} failed")
    print(f"  Timestamp: {ts}")
    print(f"{'='*60}")

    sys.exit(1 if fail > 0 else 0)

if __name__ == "__main__":
    main()
