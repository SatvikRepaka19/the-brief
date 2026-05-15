"""Fetch equity price, volume, and OHLCV data via yfinance."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

SECTOR_ETFS    = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"]
INDEX_ETFS     = ["SPY", "RSP", "QQQ", "IWM", "DIA"]
FACTOR_ETFS    = ["USMV", "MTUM", "VLUE", "QUAL", "HDV", "SIZE"]
COMMODITY_ETFS = ["UUP", "GLD", "USO", "CPER"]   # Dollar, Gold, Oil, Copper

ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + FACTOR_ETFS + COMMODITY_ETFS


def _date_range(years: int = 2) -> tuple[str, str]:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_equity_data(years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch 2 years of daily equity data for all sector, index, and factor ETFs.

    Returns
    -------
    prices_df : adjusted close prices, datetime index, columns = tickers
    volume_df : volume, same shape
    ohlc_df   : full OHLCV for SPY only (for candlestick chart)
    """
    start, end = _date_range(years)

    raw = yf.download(
        tickers=ALL_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    prices_df = raw["Close"].copy()
    volume_df = raw["Volume"].copy()

    prices_df.index = pd.to_datetime(prices_df.index)
    volume_df.index = pd.to_datetime(volume_df.index)
    prices_df.index.name = "date"
    volume_df.index.name = "date"

    prices_df = prices_df.dropna(how="all")
    volume_df = volume_df.dropna(how="all")

    ohlc_df = _fetch_spy_ohlcv(start, end)

    return prices_df, volume_df, ohlc_df


def _fetch_spy_ohlcv(start: str, end: str) -> pd.DataFrame:
    """Full OHLCV for SPY — used for the candlestick chart."""
    spy = yf.download(
        tickers="SPY",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    spy.index = pd.to_datetime(spy.index)
    spy.index.name = "date"

    # Flatten multi-level columns if present (single ticker still wraps them)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return spy


def fetch_prices(years: int = 2) -> pd.DataFrame:
    """Convenience wrapper — returns prices_df only."""
    prices_df, _, _ = fetch_equity_data(years)
    return prices_df


def fetch_returns(prices_df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """
    Compute percentage returns over multiple periods from a prices DataFrame.

    Parameters
    ----------
    periods : list of ints, trading days. Default [1, 5, 21, 252]

    Returns a MultiIndex-column DataFrame: (period_label, ticker)
    """
    if periods is None:
        periods = [1, 5, 21, 252]

    labels = {1: "1D", 5: "5D", 21: "1M", 63: "3M", 126: "6M", 252: "12M"}
    frames = {}
    for p in periods:
        label = labels.get(p, f"{p}D")
        frames[label] = prices_df.pct_change(p).iloc[-1]

    return pd.DataFrame(frames).T
