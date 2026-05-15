"""
Factor ETF relative performance, rolling alpha, sector relative performance,
and individual ETF flow-price data for dual-axis charts.
"""

import warnings
import numpy as np
import pandas as pd

from analytics.signals import rolling_zscore, SECTOR_ETFS, FACTOR_ETFS, INDEX_ETFS

# Sector groups for tilt analysis
CYCLICAL_SECTORS  = ["XLK", "XLF", "XLY", "XLI", "XLB", "XLC", "XLE"]
DEFENSIVE_SECTORS = ["XLP", "XLU", "XLV", "XLRE"]

ALL_ETF_TICKERS = SECTOR_ETFS + FACTOR_ETFS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _index_to_one(series: pd.Series) -> pd.Series:
    """Divide a price series by its first non-NaN value to index it to 1.0."""
    first = series.dropna().iloc[0] if not series.dropna().empty else np.nan
    return series / first if first and not np.isnan(first) else series


def _rolling_compound(log_ret: pd.Series, window: int) -> pd.Series:
    """Rolling compounded return over `window` periods using log-return trick (O(n))."""
    rolling_sum = log_ret.rolling(window, min_periods=window // 2).sum()
    return np.exp(rolling_sum) - 1


# ── Factor functions ──────────────────────────────────────────────────────────

def factor_relative_performance(prices_df: pd.DataFrame,
                                 base: str = "SPY") -> pd.DataFrame:
    """
    For each factor ETF: (factor_price / base_price) indexed to 1.0 at start.
    Returns DataFrame — columns = factor tickers, datetime index.
    Rising line = factor outperforming base.
    """
    if base not in prices_df.columns:
        warnings.warn(f"[factors] {base} not in prices_df — returning empty")
        return pd.DataFrame()

    spy = prices_df[base]
    result = {}
    for ticker in FACTOR_ETFS:
        if ticker not in prices_df.columns:
            continue
        ratio = (prices_df[ticker] / spy).dropna()
        result[ticker] = _index_to_one(ratio)

    return pd.DataFrame(result).dropna(how="all")


def rolling_alpha(prices_df: pd.DataFrame,
                  window: int = 126,
                  base: str = "SPY") -> pd.DataFrame:
    """
    Rolling `window`-day compounded return of each factor ETF minus SPY.
    Uses the log-return trick for O(n) efficiency (no slow .apply lambda).

    Returns DataFrame — columns = factor tickers, datetime index.
    Positive = factor beat SPY over the rolling period.
    """
    if base not in prices_df.columns:
        warnings.warn(f"[factors] {base} not in prices_df — returning empty")
        return pd.DataFrame()

    log_rets = np.log(prices_df / prices_df.shift(1))
    spy_compound = _rolling_compound(log_rets[base], window)

    result = {}
    for ticker in FACTOR_ETFS:
        if ticker not in prices_df.columns:
            continue
        factor_compound = _rolling_compound(log_rets[ticker], window)
        result[ticker] = (factor_compound - spy_compound).rename(ticker)

    return pd.DataFrame(result).dropna(how="all")


def sector_relative_performance(prices_df: pd.DataFrame,
                                 base: str = "SPY") -> dict:
    """
    For each sector ETF: (sector_price / base_price) indexed to 1.0 at start.

    Returns
    -------
    dict with keys:
        'cyclical'  — DataFrame (XLK, XLF, XLY, XLI, XLB, XLC, XLE vs base)
        'defensive' — DataFrame (XLP, XLU, XLV, XLRE vs base)
    """
    if base not in prices_df.columns:
        warnings.warn(f"[factors] {base} not in prices_df — returning empty dicts")
        return {"cyclical": pd.DataFrame(), "defensive": pd.DataFrame()}

    spy = prices_df[base]

    def _rel(tickers: list) -> pd.DataFrame:
        frames = {}
        for t in tickers:
            if t not in prices_df.columns:
                continue
            ratio = (prices_df[t] / spy).dropna()
            frames[t] = _index_to_one(ratio)
        return pd.DataFrame(frames).dropna(how="all")

    return {
        "cyclical":  _rel(CYCLICAL_SECTORS),
        "defensive": _rel(DEFENSIVE_SECTORS),
    }


def etf_flow_price(prices_df: pd.DataFrame,
                   volume_df: pd.DataFrame,
                   ticker: str,
                   window: int = 252) -> dict:
    """
    For a single ETF, return the data needed for a dual-axis flow+price chart.

    Parameters
    ----------
    window : number of most-recent trading days to include

    Returns
    -------
    dict:
        dates   — list of ISO date strings (last `window` days)
        returns — cumulative % return from window start (indexed to 0.0 %)
        flow_z  — rolling z-score of volume (green >0, red <0)
    """
    if ticker not in prices_df.columns:
        return {"dates": [], "returns": [], "flow_z": []}

    price = prices_df[ticker].dropna()

    # Compute flow_z on full history so early-window z-scores are valid
    if ticker in volume_df.columns:
        flow_z_full = rolling_zscore(volume_df[ticker].dropna(), window=window)
    else:
        flow_z_full = pd.Series(np.nan, index=price.index)

    # Slice to display window
    display_price  = price.iloc[-window:]
    display_flow_z = flow_z_full.reindex(display_price.index)

    cum_ret = (display_price / display_price.iloc[0] - 1) * 100

    def _clean(vals) -> list:
        return [None if (v != v or np.isinf(v)) else round(float(v), 4) for v in vals]

    return {
        "dates":   [d.strftime("%Y-%m-%d") for d in display_price.index],
        "returns": _clean(cum_ret.values),
        "flow_z":  _clean(display_flow_z.values),
    }


# ── Top-level aggregator ─────────────────────────────────────────────────────

def compute_all_factors(data_dict: dict) -> dict:
    """
    Entry point for Phase 2 factor analytics.

    Parameters
    ----------
    data_dict : output of cache_all()

    Returns
    -------
    dict with keys:
        factor_rel_perf  — pd.DataFrame
        factor_alpha     — pd.DataFrame
        sector_rel_perf  — dict {'cyclical': pd.DataFrame, 'defensive': pd.DataFrame}
        etf_flow_price   — dict {ticker: {dates, returns, flow_z}}
    """
    prices_df, volume_df, _ = data_dict["equity"]

    return {
        "factor_rel_perf":  factor_relative_performance(prices_df),
        "factor_alpha":     rolling_alpha(prices_df),
        "sector_rel_perf":  sector_relative_performance(prices_df),
        "etf_flow_price": {
            ticker: etf_flow_price(prices_df, volume_df, ticker)
            for ticker in ALL_ETF_TICKERS
            if ticker in prices_df.columns
        },
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(7)
    n = 320
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    tickers = SECTOR_ETFS + INDEX_ETFS + FACTOR_ETFS

    prices = pd.DataFrame(
        100 + np.random.randn(n, len(tickers)).cumsum(0),
        index=dates, columns=tickers,
    )
    volumes = pd.DataFrame(
        np.abs(np.random.randn(n, len(tickers))) * 1_000_000 + 500_000,
        index=dates, columns=tickers,
    )

    print("── factor_relative_performance ─────────────────")
    frp = factor_relative_performance(prices)
    print(f"  shape={frp.shape}  cols={list(frp.columns)}")
    print(f"  first row all ≈ 1.0: {(frp.dropna().iloc[0].round(2) == 1.0).all()}")

    print("\n── rolling_alpha (window=63) ────────────────────")
    ra = rolling_alpha(prices, window=63)
    print(f"  shape={ra.shape}  cols={list(ra.columns)}")
    print(f"  last row:\n{ra.iloc[-1].round(4)}")

    print("\n── sector_relative_performance ─────────────────")
    srp = sector_relative_performance(prices)
    print(f"  cyclical  shape={srp['cyclical'].shape}  cols={list(srp['cyclical'].columns)}")
    print(f"  defensive shape={srp['defensive'].shape} cols={list(srp['defensive'].columns)}")

    print("\n── etf_flow_price (SPY, window=63) ─────────────")
    fp = etf_flow_price(prices, volumes, "SPY", window=63)
    print(f"  dates[0]={fp['dates'][0]}  dates[-1]={fp['dates'][-1]}")
    print(f"  returns len={len(fp['returns'])}  flow_z len={len(fp['flow_z'])}")
    print(f"  returns[0]={fp['returns'][0]}  (expect 0.0)")

    print("\n── compute_all_factors ──────────────────────────")
    out = compute_all_factors({"equity": (prices, volumes, pd.DataFrame())})
    print(f"  keys: {list(out.keys())}")
    print(f"  etf_flow_price tickers: {len(out['etf_flow_price'])}")
    print("All factor checks passed.")
