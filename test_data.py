"""
Smoke test for the data pipeline.

Run:  python test_data.py

Pass 1 — cold run: fetches from FRED + yfinance, saves cache files.
Pass 2 — hot run:  loads instantly from disk, no network calls.
"""

import time
from data.cache import cache_all


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def run_test(label: str) -> dict:
    _section(label)
    t0 = time.perf_counter()
    all_data = cache_all()
    elapsed = time.perf_counter() - t0
    print(f"\n[timing] {label} completed in {elapsed:.2f}s")
    return all_data


def report_fred(fred: dict) -> None:
    _section("FRED — treasury yields")
    ty = fred["treasury_yields"]
    print(f"  shape : {ty.shape}")
    print(f"  cols  : {list(ty.columns)}")
    print(ty.tail(3).to_string())

    _section("FRED — macro series (spot check: CPI, Fed Funds, GDP)")
    for key in ("CPIAUCSL", "FEDFUNDS", "GDP"):
        df = fred["macro_series"][key]
        print(f"\n  {key}  shape={df.shape}  last={df.index[-1].date()}  value={df.iloc[-1, 0]:.4f}")

    _section("FRED — yield curve spreads")
    df = fred["yield_curve_spreads"]
    print(f"  shape : {df.shape}")
    print(df.tail(3).to_string())

    _section("FRED — credit spreads")
    df = fred["credit_spreads"]
    print(f"  shape : {df.shape}")
    print(df.tail(3).to_string())


def report_equity(equity: tuple) -> None:
    prices_df, volume_df, ohlc_df = equity

    _section("EQUITY — prices_df")
    print(f"  shape : {prices_df.shape}")
    print(f"  cols  : {list(prices_df.columns)}")
    print(prices_df.tail(3).to_string())

    _section("EQUITY — no NaN in last row of prices_df")
    last_row = prices_df.iloc[-1]
    nan_cols = last_row[last_row.isna()].index.tolist()
    if nan_cols:
        print(f"  WARN — NaN in last row for: {nan_cols}")
    else:
        print("  OK — no NaN values in last row")

    _section("EQUITY — volume_df")
    print(f"  shape : {volume_df.shape}")
    print(volume_df.tail(3).to_string())

    _section("EQUITY — SPY OHLCV (ohlc_df)")
    print(f"  shape : {ohlc_df.shape}")
    print(f"  cols  : {list(ohlc_df.columns)}")
    print(ohlc_df.tail(3).to_string())


if __name__ == "__main__":
    # --- Pass 1: cold (or cache hit if already run today) ---
    data1 = run_test("Pass 1")
    report_fred(data1["fred"])
    report_equity(data1["equity"])

    # --- Pass 2: should be instant cache hit ---
    print("\n\n" + "=" * 60)
    print("  Pass 2 — expect instant cache hits")
    print("=" * 60)
    t0 = time.perf_counter()
    data2 = cache_all()  # noqa: F841  (intentional re-call to verify cache)
    elapsed = time.perf_counter() - t0
    print(f"\n[timing] Pass 2 completed in {elapsed:.3f}s  (should be <0.5s)")
    assert elapsed < 5.0, "Cache hit took too long — something is re-fetching"
    print("\nAll checks passed.")
