"""
The Brief — master generation script.
Fetch → Analyze → Serialize → Write data.json.
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.cache import cache_all
from analytics.signals import (
    compute_all_signals,
    market_regime_score,
    correlation_matrix,
    fifty_two_week_range,
    drawdown_series,
    commodities_snapshot,
)
from analytics.factors import compute_all_factors
from analytics.fixed_income import compute_all_fixed_income, inflation_decomposition


# ── JSON serialization ────────────────────────────────────────────────────────

def _float_or_none(v) -> float | None:
    """Convert any numeric-ish value to a clean Python float, or None for NaN/inf."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _clean_list(arr) -> list:
    """Convert an iterable to a list of clean Python float | int | None."""
    result = []
    for v in arr:
        if isinstance(v, (np.floating,)):
            fv = float(v)
            result.append(None if (math.isnan(fv) or math.isinf(fv)) else fv)
        elif isinstance(v, float):
            result.append(None if (math.isnan(v) or math.isinf(v)) else v)
        elif isinstance(v, (np.integer,)):
            result.append(int(v))
        elif isinstance(v, int):
            result.append(v)
        elif v is None:
            result.append(None)
        else:
            result.append(v)
    return result


def _date_list(index: pd.Index) -> list[str]:
    """Convert a pandas Index to a list of ISO date strings."""
    return [
        d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d)
        for d in index
    ]


def _series_to_json(s: pd.Series) -> dict:
    """Convert a pandas Series (datetime index) to {dates, values}."""
    return {"dates": _date_list(s.index), "values": _clean_list(s.values)}


def _df_to_json(df: pd.DataFrame) -> dict | list:
    """
    Convert a pandas DataFrame to a JSON-friendly structure.
    DatetimeIndex → {dates, series}.  Other index → list of records.
    """
    if df.empty:
        if isinstance(df.index, pd.DatetimeIndex):
            return {"dates": [], "series": {col: [] for col in df.columns}}
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        return {
            "dates":  _date_list(df.index),
            "series": {str(col): _clean_list(df[col].values) for col in df.columns},
        }
    # Non-datetime index (sector composite table etc.) → records
    records = df.to_dict(orient="records")
    return [_sanitize(rec) for rec in records]


def _sanitize(obj):
    """
    Recursively convert an object tree to JSON-safe Python primitives.
    Handles: dict, list, pd.DataFrame, pd.Series, pd.Timestamp,
             np.integer, np.floating, float NaN/inf, np.ndarray.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _df_to_json(obj)
    if isinstance(obj, pd.Series):
        return _series_to_json(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _float_or_none(float(obj))
    if isinstance(obj, float):
        return _float_or_none(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ── Phase 3 helper builders ───────────────────────────────────────────────────

def _build_index_returns(prices_df: pd.DataFrame) -> dict:
    """
    Full-history cumulative returns (%) for SPY, QQQ, IWM, DIA.
    Frontend uses these to rebase to any period start.
    """
    result = {}
    for t in ["SPY", "QQQ", "IWM", "DIA"]:
        if t not in prices_df.columns:
            continue
        price = prices_df[t].dropna()
        if price.empty:
            continue
        cum = (price / price.iloc[0] - 1) * 100
        result[t] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in cum.index],
            "values": _clean_list(cum.values),
        }
    return result


def _build_index_snapshot(prices_df: pd.DataFrame) -> list[dict]:
    """Latest price + 1D % for index ETFs — used in the ticker bar."""
    result = []
    for t in ["SPY", "QQQ", "IWM", "DIA"]:
        if t not in prices_df.columns:
            continue
        price = prices_df[t].dropna()
        if len(price) < 2:
            continue
        result.append({
            "ticker": t,
            "price":  round(float(price.iloc[-1]), 2),
            "1D":     round(float(price.pct_change(1).iloc[-1]) * 100, 2),
        })
    return result


# ── Output assembly ───────────────────────────────────────────────────────────

def _last_trading_day(prices_df: pd.DataFrame) -> str:
    try:
        return prices_df.dropna(how="all").index[-1].strftime("%Y-%m-%d")
    except Exception:
        return datetime.today().strftime("%Y-%m-%d")


def build_output(data: dict, signals: dict, factors: dict, fi: dict) -> dict:
    prices_df, _, _ = data["equity"]
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_date":    _last_trading_day(prices_df),
        },
        "equities":     signals,
        "factors":      factors,
        "fixed_income": fi,
    }


def write_json(output: dict, path: str = "data.json") -> None:
    sanitized = _sanitize(output)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sanitized, fh, indent=2, ensure_ascii=False)
    size_kb = os.path.getsize(path) / 1024
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"data.json written — {ts} — {size_kb:.1f} kb")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("The Brief — generating data.json")
    print("=" * 50)

    print("[1/5] Loading data …")
    data = cache_all()
    prices_df, volume_df, ohlc_df = data["equity"]
    fred_data = data["fred"]
    print(f"      equity : {prices_df.shape[1]} tickers × {prices_df.shape[0]} days")
    print(f"      FRED   : {len(fred_data['macro_series'])} macro series, "
          f"{fred_data['treasury_yields'].shape[1] if not fred_data['treasury_yields'].empty else 0} tenors")

    print("[2/5] Computing equity signals …")
    signals = compute_all_signals(data)

    # ── Phase 3 equity additions ──────────────────────────────────────────────
    signals["market_regime"]   = market_regime_score(signals)
    signals["correlation"]     = correlation_matrix(prices_df)
    signals["fiftytwo_week"]   = fifty_two_week_range(prices_df)
    signals["drawdown"]        = drawdown_series(prices_df)
    signals["commodities"]     = commodities_snapshot(prices_df)
    signals["index_returns"]   = _build_index_returns(prices_df)
    signals["index_snapshot"]  = _build_index_snapshot(prices_df)
    signals["spy_ohlcv"]       = ohlc_df          # DataFrame → _df_to_json via _sanitize
    # ─────────────────────────────────────────────────────────────────────────

    print(f"      market_regime  : {signals['market_regime']['score']} ({signals['market_regime']['label']})")
    print(f"      fiftytwo_week  : {len(signals['fiftytwo_week'])} sectors")
    print(f"      drawdown       : {len(signals['drawdown'])} tickers")
    print(f"      commodities    : {len(signals['commodities']['snapshot'])} tickers")

    print("[3/5] Computing factor analytics …")
    factors = compute_all_factors(data)
    print(f"      factor_rel_perf  : {factors['factor_rel_perf'].shape}")
    print(f"      etf_flow_price   : {len(factors['etf_flow_price'])} tickers")

    print("[4/5] Computing fixed income analytics …")
    fi = compute_all_fixed_income(data)
    # ── Phase 3 fixed income addition ────────────────────────────────────────
    fi["inflation_decomp"] = inflation_decomposition(fred_data)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"      yield_curve     : {fi['yield_curve_snapshot']['shape']}")
    print(f"      gdp bars        : {len(fi['gdp_growth']['dates'])}")
    infl_cols = list(fi["inflation_decomp"]["data"].columns) if not fi["inflation_decomp"]["data"].empty else []
    print(f"      inflation_decomp: {len(infl_cols)} components — {infl_cols}")

    print("[5/5] Serializing and writing data.json …")
    output = build_output(data, signals, factors, fi)
    write_json(output)
    print("Done.")


if __name__ == "__main__":
    main()
