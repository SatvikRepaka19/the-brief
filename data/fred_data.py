"""Fetch macroeconomic series from the FRED API."""

import os
import warnings
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

TREASURY_YIELDS = [
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS5", "DGS10", "DGS20", "DGS30",
]

MACRO_SERIES = {
    "CPIAUCSL":     "CPI",
    "CPILFESL":     "Core CPI",
    "PCEPI":        "PCE",
    "PCEPILFE":     "Core PCE",
    "FEDFUNDS":     "Fed Funds Rate",
    "UNRATE":       "Unemployment Rate",
    "GDP":          "Real GDP",
    "INDPRO":       "Industrial Production",
    "RSAFS":        "Retail Sales",
    "T10Y2Y":       "10Y-2Y Spread",
    "T10Y3M":       "10Y-3M Spread",
    "DFII10":       "10Y Real Yield",
    "T10YIE":       "10Y Breakeven Inflation",
    "BAMLH0A0HYM2": "HY OAS",
    "BAMLC0A0CM":   "IG OAS",
}

# Weekly fallbacks for series known to have intermittent FRED 500 errors
_FALLBACK_IDS = {
    "DFII10": "WDFII10",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_fred() -> Fred:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise EnvironmentError("FRED_API_KEY not set in environment / .env file")
    return Fred(api_key=key)


def _clean(series: pd.Series, name: str) -> pd.DataFrame:
    """Convert a FRED Series to a tidy DataFrame: datetime index, no trailing NaNs."""
    df = series.to_frame(name=name)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.dropna()


def _empty_df(*columns: str) -> pd.DataFrame:
    """Return an empty DataFrame with a datetime index and the given columns."""
    df = pd.DataFrame(columns=list(columns))
    df.index = pd.DatetimeIndex([], name="date")
    return df


def _fetch_series(fred: Fred, series_id: str, start: str) -> pd.Series:
    """
    Fetch a single FRED series, transparently trying a weekly fallback ID if the
    primary series returns a 500 error.  Raises on all other failures.
    """
    try:
        return fred.get_series(series_id, observation_start=start)
    except Exception as primary_err:
        fallback = _FALLBACK_IDS.get(series_id)
        if fallback:
            warnings.warn(
                f"[fred] {series_id} failed ({primary_err}); "
                f"retrying with fallback {fallback}",
                stacklevel=2,
            )
            return fred.get_series(fallback, observation_start=start)
        raise


# ---------------------------------------------------------------------------
# Public fetch functions — each returns an empty DataFrame on any failure
# so a single bad series never kills cache_all()
# ---------------------------------------------------------------------------

def fetch_treasury_yields(start: str = "2000-01-01") -> pd.DataFrame:
    """Return a wide DataFrame of all treasury yield series (columns = tickers)."""
    try:
        fred = _get_fred()
        frames = {}
        for ticker in TREASURY_YIELDS:
            try:
                raw = _fetch_series(fred, ticker, start)
                frames[ticker] = raw.dropna()
            except Exception as e:
                warnings.warn(f"[fred] treasury yield {ticker} skipped: {e}", stacklevel=2)

        if not frames:
            return _empty_df(*TREASURY_YIELDS)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(how="all")
        df = df[: df.last_valid_index()]
        return df
    except Exception as e:
        warnings.warn(f"[fred] fetch_treasury_yields failed entirely: {e}", stacklevel=2)
        return _empty_df(*TREASURY_YIELDS)


def fetch_macro_series(start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    """Return a dict of {series_id: DataFrame} for every macro series."""
    fred = _get_fred()
    result = {}
    for series_id, label in MACRO_SERIES.items():
        try:
            raw = _fetch_series(fred, series_id, start)
            result[series_id] = _clean(raw, label)
        except Exception as e:
            warnings.warn(f"[fred] macro series {series_id} skipped: {e}", stacklevel=2)
            result[series_id] = _empty_df(label)
    return result


def fetch_cpi(start: str = "2000-01-01") -> pd.DataFrame:
    """CPI (CPIAUCSL) and Core CPI (CPILFESL) in one DataFrame."""
    try:
        fred = _get_fred()
        cpi  = _fetch_series(fred, "CPIAUCSL", start).dropna()
        core = _fetch_series(fred, "CPILFESL", start).dropna()
        df = pd.DataFrame({"CPI": cpi, "Core CPI": core})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna()
    except Exception as e:
        warnings.warn(f"[fred] fetch_cpi failed: {e}", stacklevel=2)
        return _empty_df("CPI", "Core CPI")


def fetch_pce(start: str = "2000-01-01") -> pd.DataFrame:
    """PCE (PCEPI) and Core PCE (PCEPILFE) in one DataFrame."""
    try:
        fred = _get_fred()
        pce  = _fetch_series(fred, "PCEPI",    start).dropna()
        core = _fetch_series(fred, "PCEPILFE", start).dropna()
        df = pd.DataFrame({"PCE": pce, "Core PCE": core})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna()
    except Exception as e:
        warnings.warn(f"[fred] fetch_pce failed: {e}", stacklevel=2)
        return _empty_df("PCE", "Core PCE")


def fetch_yield_curve_spreads(start: str = "2000-01-01") -> pd.DataFrame:
    """10Y-2Y and 10Y-3M spreads in one DataFrame."""
    try:
        fred   = _get_fred()
        t10y2y = _fetch_series(fred, "T10Y2Y", start).dropna()
        t10y3m = _fetch_series(fred, "T10Y3M", start).dropna()
        df = pd.DataFrame({"10Y-2Y": t10y2y, "10Y-3M": t10y3m})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna(how="all")
    except Exception as e:
        warnings.warn(f"[fred] fetch_yield_curve_spreads failed: {e}", stacklevel=2)
        return _empty_df("10Y-2Y", "10Y-3M")


def fetch_real_yield_breakeven(start: str = "2000-01-01") -> pd.DataFrame:
    """
    10Y Real Yield and 10Y Breakeven Inflation.

    Tries DFII10 (daily) first; falls back to WDFII10 (weekly) if FRED returns a
    500 error.  Returns an empty DataFrame if both fail so the pipeline keeps running.
    """
    try:
        fred = _get_fred()

        # Real yield — primary DFII10, fallback WDFII10
        try:
            real = _fetch_series(fred, "DFII10", start).dropna()
        except Exception as e:
            warnings.warn(
                f"[fred] DFII10 and WDFII10 both failed: {e}; "
                "Real Yield column will be empty",
                stacklevel=2,
            )
            real = pd.Series(dtype=float, name="Real Yield 10Y")

        # Breakeven — no known fallback, but isolate its failure
        try:
            be = _fetch_series(fred, "T10YIE", start).dropna()
        except Exception as e:
            warnings.warn(
                f"[fred] T10YIE failed: {e}; Breakeven column will be empty",
                stacklevel=2,
            )
            be = pd.Series(dtype=float, name="Breakeven 10Y")

        df = pd.DataFrame({"Real Yield 10Y": real, "Breakeven 10Y": be})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna(how="all")

    except Exception as e:
        warnings.warn(f"[fred] fetch_real_yield_breakeven failed entirely: {e}", stacklevel=2)
        return _empty_df("Real Yield 10Y", "Breakeven 10Y")


def fetch_credit_spreads(start: str = "2000-01-01") -> pd.DataFrame:
    """HY OAS (BAMLH0A0HYM2) and IG OAS (BAMLC0A0CM)."""
    try:
        fred = _get_fred()
        hy = _fetch_series(fred, "BAMLH0A0HYM2", start).dropna()
        ig = _fetch_series(fred, "BAMLC0A0CM",   start).dropna()
        df = pd.DataFrame({"HY OAS": hy, "IG OAS": ig})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna(how="all")
    except Exception as e:
        warnings.warn(f"[fred] fetch_credit_spreads failed: {e}", stacklevel=2)
        return _empty_df("HY OAS", "IG OAS")


def fetch_fed_funds_unemployment(start: str = "2000-01-01") -> pd.DataFrame:
    """Fed Funds Rate and Unemployment Rate together."""
    try:
        fred = _get_fred()
        ff = _fetch_series(fred, "FEDFUNDS", start).dropna()
        ur = _fetch_series(fred, "UNRATE",   start).dropna()
        df = pd.DataFrame({"Fed Funds Rate": ff, "Unemployment Rate": ur})
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.dropna(how="all")
    except Exception as e:
        warnings.warn(f"[fred] fetch_fed_funds_unemployment failed: {e}", stacklevel=2)
        return _empty_df("Fed Funds Rate", "Unemployment Rate")


def fetch_gdp(start: str = "2000-01-01") -> pd.DataFrame:
    """Real GDP (quarterly)."""
    try:
        fred = _get_fred()
        raw = _fetch_series(fred, "GDP", start)
        return _clean(raw, "Real GDP")
    except Exception as e:
        warnings.warn(f"[fred] fetch_gdp failed: {e}", stacklevel=2)
        return _empty_df("Real GDP")


# CPI sub-components for inflation decomposition
CPI_SUB_COMPONENTS = {
    "CUSR0000SAH1":   "Shelter",
    "CUSR0000SA0E":   "Energy",
    "CUSR0000SAF11":  "Food at Home",
    "CUSR0000SACL1E": "Core Goods",
    "CUSR0000SASLE":  "Services",
}


def fetch_cpi_components(start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    """
    Fetch CPI sub-component series for inflation decomposition chart.
    Returns dict of {series_id: DataFrame}.  Each series fails independently.
    """
    fred = _get_fred()
    result = {}
    for series_id, label in CPI_SUB_COMPONENTS.items():
        try:
            raw = _fetch_series(fred, series_id, start)
            result[series_id] = _clean(raw, label)
        except Exception as e:
            warnings.warn(f"[fred] CPI component {series_id} skipped: {e}", stacklevel=2)
            result[series_id] = _empty_df(label)
    return result


def fetch_all_fred(start: str = "2000-01-01") -> dict:
    """
    Fetch everything and return a single dict with keys:
      treasury_yields, macro_series, cpi, pce, yield_curve_spreads,
      real_yield_breakeven, credit_spreads, fed_funds_unemployment, gdp,
      cpi_components

    Each value is either a populated DataFrame/dict or a safe empty placeholder —
    no individual series failure will raise here.
    """
    return {
        "treasury_yields":        fetch_treasury_yields(start),
        "macro_series":           fetch_macro_series(start),
        "cpi":                    fetch_cpi(start),
        "pce":                    fetch_pce(start),
        "yield_curve_spreads":    fetch_yield_curve_spreads(start),
        "real_yield_breakeven":   fetch_real_yield_breakeven(start),
        "credit_spreads":         fetch_credit_spreads(start),
        "fed_funds_unemployment": fetch_fed_funds_unemployment(start),
        "gdp":                    fetch_gdp(start),
        "cpi_components":         fetch_cpi_components(start),
    }
