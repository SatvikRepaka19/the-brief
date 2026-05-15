"""Daily pickle cache for FRED and equity data."""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from data.fred_data import fetch_all_fred
from data.equity_data import fetch_equity_data

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_MAX_AGE_DAYS = 3

# Schema versioning — add keys here when fetch_all_fred gains new series
REQUIRED_FRED_KEYS = [
    "treasury_yields", "macro_series", "cpi", "pce",
    "yield_curve_spreads", "real_yield_breakeven", "credit_spreads",
    "fed_funds_unemployment", "gdp", "cpi_components",
]
# Tickers that must be present in prices_df for equity cache to be valid
REQUIRED_EQUITY_TICKERS = ["UUP", "GLD", "USO"]


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_{_today()}.pkl"


def _purge_old_files() -> None:
    """Delete cache files older than CACHE_MAX_AGE_DAYS."""
    cutoff = datetime.today() - timedelta(days=CACHE_MAX_AGE_DAYS)
    for f in CACHE_DIR.glob("*.pkl"):
        try:
            # filename format: <name>_YYYY-MM-DD.pkl
            date_str = f.stem.rsplit("_", 1)[-1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                f.unlink()
        except (ValueError, IndexError):
            pass  # skip files that don't match the naming pattern


def _load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save(path: Path, obj) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def cache_fred(start: str = "2000-01-01") -> dict:
    """
    Return FRED data dict, loading from today's cache if available.
    Treats cache as stale if required keys are missing (schema evolution).
    """
    path = _cache_path("fred")
    if path.exists():
        data = _load(path)
        if all(k in data for k in REQUIRED_FRED_KEYS):
            print(f"[cache] HIT  {path.name}")
            return data
        print("[cache] STALE fred — missing new keys, re-fetching …")

    print("[cache] MISS fred — fetching from FRED API …")
    data = fetch_all_fred(start)
    _save(path, data)
    _purge_old_files()
    print(f"[cache] SAVED {path.name}")
    return data


def cache_equity(years: int = 2):
    """
    Return (prices_df, volume_df, ohlc_df), loading from today's cache if available.
    Treats cache as stale if required commodity tickers are missing.
    """
    path = _cache_path("equity")
    if path.exists():
        data = _load(path)
        prices_df, _, _ = data
        if all(t in prices_df.columns for t in REQUIRED_EQUITY_TICKERS):
            print(f"[cache] HIT  {path.name}")
            return data
        print("[cache] STALE equity — missing new tickers, re-fetching …")

    print("[cache] MISS equity — fetching from yfinance …")
    data = fetch_equity_data(years)
    _save(path, data)
    _purge_old_files()
    print(f"[cache] SAVED {path.name}")
    return data


def cache_all(fred_start: str = "2000-01-01", equity_years: int = 2) -> dict:
    """
    Fetch (or load) both FRED and equity data.

    Returns
    -------
    {
        "fred":   { ... },              # output of fetch_all_fred()
        "equity": (prices, volume, ohlc) # output of fetch_equity_data()
    }
    """
    return {
        "fred":   cache_fred(fred_start),
        "equity": cache_equity(equity_years),
    }
