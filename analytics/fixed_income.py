"""
Fixed income and macro analytics: yield curve, spreads, inflation, credit, GDP.
All public functions accept fred_data (the 'fred' sub-dict from cache_all()).
"""

import warnings
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

TENOR_MAP: OrderedDict = OrderedDict([
    ("1M",  "DGS1MO"),
    ("3M",  "DGS3MO"),
    ("6M",  "DGS6MO"),
    ("1Y",  "DGS1"),
    ("2Y",  "DGS2"),
    ("5Y",  "DGS5"),
    ("10Y", "DGS10"),
    ("20Y", "DGS20"),
    ("30Y", "DGS30"),
])

PERIOD_CUTOFFS = {
    "1Y":   365,
    "3Y":   3 * 365,
    "5Y":   5 * 365,
    "10Y":  10 * 365,
    "Full": None,
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    """Convert to float, returning None for NaN / inf."""
    try:
        f = float(v)
        return None if (f != f or f == float("inf") or f == float("-inf")) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _series_last(series: pd.Series, n: int = 1):
    """Return the nth-from-last valid value, or None."""
    clean = series.dropna()
    if len(clean) < n:
        return None
    return _safe_float(clean.iloc[-n])


def _classify_curve_shape(y2: float | None,
                           y10: float | None,
                           y30: float | None) -> str:
    """Classify yield curve shape from key spot rates."""
    if any(v is None for v in [y2, y10, y30]):
        return "unknown"
    spread_2_10  = y10 - y2
    spread_10_30 = y30 - y10
    if spread_2_10 < -0.10:
        return "inverted"
    if abs(spread_2_10) <= 0.25:
        return "flat"
    if spread_2_10 > 0.25 and spread_10_30 < -0.10:
        return "humped"
    return "upward sloping"


def _find_inversion_periods(series: pd.Series) -> list[dict]:
    """
    Return list of {start, end} dicts for contiguous periods where series < 0.
    Dates are ISO strings.
    """
    periods = []
    in_inv, start = False, None
    for date, val in series.items():
        if pd.isna(val):
            continue
        if val < 0 and not in_inv:
            in_inv, start = True, date
        elif val >= 0 and in_inv:
            in_inv = False
            periods.append({"start": start.strftime("%Y-%m-%d"),
                            "end":   date.strftime("%Y-%m-%d")})
    if in_inv and start is not None:
        periods.append({"start": start.strftime("%Y-%m-%d"),
                        "end":   series.index[-1].strftime("%Y-%m-%d")})
    return periods


def _filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Slice a DataFrame to the requested look-back period."""
    days = PERIOD_CUTOFFS.get(period)
    if days is None:
        return df
    cutoff = pd.Timestamp(datetime.today() - timedelta(days=days))
    return df[df.index >= cutoff]


def _yoy(series: pd.Series, periods: int = 12) -> pd.Series:
    """Compute year-over-year % change from a levels series."""
    return series.pct_change(periods) * 100


# ── Public functions ──────────────────────────────────────────────────────────

def yield_curve_snapshot(fred_data: dict) -> dict:
    """
    Latest spot rates and daily changes for all tenors.
    Classifies curve shape and generates a human-readable summary string.

    Returns
    -------
    dict:
        tenors  — ordered list of tenor labels
        rates   — {tenor: float | None}
        changes — {tenor: float | None}  (today vs previous observation)
        shape   — 'upward sloping' | 'inverted' | 'flat' | 'humped' | 'unknown'
        summary — one-sentence description string
    """
    ty = fred_data.get("treasury_yields", pd.DataFrame())
    if ty.empty or len(ty) < 2:
        return {
            "tenors": list(TENOR_MAP.keys()),
            "rates": {t: None for t in TENOR_MAP},
            "changes": {t: None for t in TENOR_MAP},
            "shape": "unavailable",
            "summary": "Treasury yield data unavailable.",
        }

    latest = ty.iloc[-1]
    prev   = ty.iloc[-2]

    rates, changes = {}, {}
    for tenor, sid in TENOR_MAP.items():
        if sid in ty.columns:
            rates[tenor]   = _safe_float(latest[sid])
            changes[tenor] = _safe_float(latest[sid] - prev[sid]) if not pd.isna(prev[sid]) else None
        else:
            rates[tenor] = changes[tenor] = None

    y2, y10, y30 = rates.get("2Y"), rates.get("10Y"), rates.get("30Y")
    shape = _classify_curve_shape(y2, y10, y30)

    if all(v is not None for v in [y2, y10, y30]):
        spread_bps = round((y10 - y2) * 100)
        spread_str = f"{'+' if spread_bps >= 0 else ''}{spread_bps}bps"
        summary = (
            f"Currently {shape} — "
            f"2Y {y2:.2f}% / 10Y {y10:.2f}% / 30Y {y30:.2f}% / "
            f"10Y-2Y {spread_str}"
        )
    else:
        summary = f"Yield curve is {shape}; some tenors unavailable."

    return {
        "tenors":  list(rates.keys()),
        "rates":   rates,
        "changes": changes,
        "shape":   shape,
        "summary": summary,
    }


def treasury_yields_history(fred_data: dict, period: str = "Full") -> pd.DataFrame:
    """
    Time-filtered DataFrame of 2Y, 5Y, 10Y, 30Y treasury yields.
    period: '1Y' | '3Y' | '5Y' | '10Y' | 'Full'
    Returns DataFrame with datetime index; period filtering is best done
    client-side — this function returns Full by default for data.json.
    """
    ty = fred_data.get("treasury_yields", pd.DataFrame())
    if ty.empty:
        return pd.DataFrame()

    col_map = {"DGS2": "2Y", "DGS5": "5Y", "DGS10": "10Y", "DGS30": "30Y"}
    cols = [c for c in col_map if c in ty.columns]
    df   = ty[cols].rename(columns=col_map).dropna(how="all")
    return _filter_period(df, period)


def key_rates_snapshot(fred_data: dict) -> list[dict]:
    """
    Latest value + daily change badge for key macro rates.
    Returns list of dicts: {label, value, change, direction, suffix}.
    direction: 'up' | 'down' | 'flat'
    """
    ty      = fred_data.get("treasury_yields",     pd.DataFrame())
    macro   = fred_data.get("macro_series",         {})
    spreads = fred_data.get("yield_curve_spreads",  pd.DataFrame())
    cpi_df  = fred_data.get("cpi",                  pd.DataFrame())

    snapshot = []

    def _badge(series: pd.Series, label: str, suffix: str = "%",
                threshold: float = 0.001) -> dict | None:
        clean = series.dropna()
        if len(clean) < 2:
            return None
        val    = _safe_float(clean.iloc[-1])
        change = _safe_float(clean.iloc[-1] - clean.iloc[-2])
        if val is None:
            return None
        direction = "up" if (change or 0) > threshold else (
                    "down" if (change or 0) < -threshold else "flat")
        return {"label": label, "value": val, "change": change,
                "direction": direction, "suffix": suffix}

    # Treasury spot rates
    for sid, label in [("DGS2", "2Y Treasury"), ("DGS10", "10Y Treasury"), ("DGS30", "30Y Treasury")]:
        if not ty.empty and sid in ty.columns:
            b = _badge(ty[sid], label)
            if b:
                snapshot.append(b)

    # Fed Funds (monthly — change vs previous month)
    ff_df = macro.get("FEDFUNDS", pd.DataFrame())
    if not ff_df.empty:
        b = _badge(ff_df.iloc[:, 0], "Fed Funds Rate")
        if b:
            snapshot.append(b)

    # CPI YoY
    if not cpi_df.empty and "CPI" in cpi_df.columns:
        cpi_yoy = _yoy(cpi_df["CPI"]).dropna()
        b = _badge(cpi_yoy, "CPI YoY", threshold=0.05)
        if b:
            snapshot.append(b)

    # 10Y-2Y spread
    if not spreads.empty and "10Y-2Y" in spreads.columns:
        b = _badge(spreads["10Y-2Y"], "10Y-2Y Spread")
        if b:
            snapshot.append(b)

    return snapshot


def curve_spreads(fred_data: dict) -> dict:
    """
    10Y-2Y and 10Y-3M spread time series with inversion period metadata.

    Returns
    -------
    dict:
        data               — pd.DataFrame (datetime index, two columns)
        inversion_10y2y    — list of {start, end} dicts
        inversion_10y3m    — list of {start, end} dicts
    """
    spreads = fred_data.get("yield_curve_spreads", pd.DataFrame())
    if spreads.empty:
        return {"data": pd.DataFrame(), "inversion_10y2y": [], "inversion_10y3m": []}

    inv_2y  = _find_inversion_periods(spreads["10Y-2Y"])  if "10Y-2Y" in spreads.columns else []
    inv_3m  = _find_inversion_periods(spreads["10Y-3M"])  if "10Y-3M" in spreads.columns else []

    return {
        "data":            spreads,
        "inversion_10y2y": inv_2y,
        "inversion_10y3m": inv_3m,
    }


def real_yield_breakeven(fred_data: dict) -> dict:
    """
    10Y Real Yield (DFII10 / WDFII10 fallback) and 10Y Breakeven Inflation (T10YIE).
    Handles empty gracefully if DFII10 failed in fetch layer.

    Returns dict:
        data — pd.DataFrame (datetime index)  [may have only one column]
    """
    ry_df = fred_data.get("real_yield_breakeven", pd.DataFrame())
    return {"data": ry_df}


def credit_spreads(fred_data: dict) -> dict:
    """
    HY OAS, IG OAS, and HY-IG gap time series.

    Returns
    -------
    dict:
        data    — pd.DataFrame with columns HY OAS, IG OAS, HY-IG Gap
        note    — 'wider = risk-off'
    """
    cs = fred_data.get("credit_spreads", pd.DataFrame())
    if cs.empty:
        return {"data": pd.DataFrame(), "note": "wider = risk-off"}

    df = cs.copy()
    if "HY OAS" in df.columns and "IG OAS" in df.columns:
        df["HY-IG Gap"] = df["HY OAS"] - df["IG OAS"]

    return {"data": df, "note": "wider = risk-off"}


def inflation_series(fred_data: dict) -> dict:
    """
    CPI YoY, Core CPI YoY, PCE YoY, Core PCE YoY from monthly FRED level data.
    Includes a 2 % target reference line value.

    Returns
    -------
    dict:
        data   — pd.DataFrame with four YoY columns and datetime index
        target — 2.0  (the Fed's 2 % inflation target)
    """
    cpi_df = fred_data.get("cpi", pd.DataFrame())
    pce_df = fred_data.get("pce", pd.DataFrame())

    frames = {}
    if not cpi_df.empty:
        if "CPI" in cpi_df.columns:
            frames["CPI YoY"]      = _yoy(cpi_df["CPI"])
        if "Core CPI" in cpi_df.columns:
            frames["Core CPI YoY"] = _yoy(cpi_df["Core CPI"])
    if not pce_df.empty:
        if "PCE" in pce_df.columns:
            frames["PCE YoY"]      = _yoy(pce_df["PCE"])
        if "Core PCE" in pce_df.columns:
            frames["Core PCE YoY"] = _yoy(pce_df["Core PCE"])

    if not frames:
        return {"data": pd.DataFrame(), "target": 2.0}

    df = pd.DataFrame(frames).dropna(how="all")
    return {"data": df, "target": 2.0}


def fed_and_labor(fred_data: dict) -> dict:
    """
    Fed Funds rate and Unemployment rate histories.

    Returns
    -------
    dict:
        data — pd.DataFrame with 'Fed Funds Rate' and 'Unemployment Rate' columns
    """
    ff_df = fred_data.get("fed_funds_unemployment", pd.DataFrame())
    return {"data": ff_df}


def gdp_growth(fred_data: dict) -> dict:
    """
    Real GDP QoQ annualized growth rate (from quarterly FRED levels).
    Each bar coloured green (positive) or red (negative).

    Returns
    -------
    dict:
        dates  — list of ISO date strings
        values — list of float | None
        colors — list of 'green' | 'red' | 'gray' (for NaN)
    """
    gdp_df = fred_data.get("gdp", pd.DataFrame())
    if gdp_df.empty:
        return {"dates": [], "values": [], "colors": []}

    gdp = gdp_df.iloc[:, 0].dropna()

    # QoQ annualized: ((1 + qoq)^4 - 1) * 100
    qoq_ann = ((gdp / gdp.shift(1)) ** 4 - 1) * 100
    qoq_ann = qoq_ann.dropna()

    dates  = [d.strftime("%Y-%m-%d") for d in qoq_ann.index]
    values = [_safe_float(v) for v in qoq_ann.values]
    colors = [
        "green" if v is not None and v >= 0 else ("red" if v is not None else "gray")
        for v in values
    ]

    return {"dates": dates, "values": values, "colors": colors}


def macro_pulse(fred_data: dict) -> dict:
    """
    Three macro series aligned on a common date index for a single combined chart:
      10Y-2Y spread, Fed Funds rate, Unemployment rate.
    Monthly series are forward-filled to the daily grid.

    Returns
    -------
    dict:
        dates        — list of ISO date strings
        spread_10y2y — list of float | None
        fed_funds    — list of float | None
        unemployment — list of float | None
    """
    spreads  = fred_data.get("yield_curve_spreads",   pd.DataFrame())
    macro    = fred_data.get("macro_series",           {})
    ff_df    = macro.get("FEDFUNDS",  pd.DataFrame())
    ur_df    = macro.get("UNRATE",    pd.DataFrame())

    # Use the 10Y-2Y daily index as the primary grid
    if not spreads.empty and "10Y-2Y" in spreads.columns:
        spread_s = spreads["10Y-2Y"]
        idx = spread_s.index
    else:
        spread_s = pd.Series(dtype=float)
        idx = pd.DatetimeIndex([])

    def _align(df: pd.DataFrame) -> pd.Series:
        if df.empty or idx.empty:
            return pd.Series(np.nan, index=idx)
        return df.iloc[:, 0].reindex(idx, method="ffill")

    ff_aligned = _align(ff_df)
    ur_aligned = _align(ur_df)

    dates = [d.strftime("%Y-%m-%d") for d in idx]

    def _to_list(s: pd.Series) -> list:
        return [_safe_float(v) for v in s.values]

    return {
        "dates":        dates,
        "spread_10y2y": _to_list(spread_s.reindex(idx)),
        "fed_funds":    _to_list(ff_aligned),
        "unemployment": _to_list(ur_aligned),
    }


# ── Top-level aggregator ──────────────────────────────────────────────────────

def compute_all_fixed_income(data_dict: dict) -> dict:
    """
    Entry point for Phase 2 fixed income analytics.

    Parameters
    ----------
    data_dict : output of cache_all()

    Returns
    -------
    dict with keys matching each function above.
    """
    fred = data_dict["fred"]
    return {
        "yield_curve_snapshot":   yield_curve_snapshot(fred),
        "treasury_yields_history": treasury_yields_history(fred, period="Full"),
        "key_rates_snapshot":     key_rates_snapshot(fred),
        "curve_spreads":          curve_spreads(fred),
        "real_yield_breakeven":   real_yield_breakeven(fred),
        "credit_spreads":         credit_spreads(fred),
        "inflation_series":       inflation_series(fred),
        "fed_and_labor":          fed_and_labor(fred),
        "gdp_growth":             gdp_growth(fred),
        "macro_pulse":            macro_pulse(fred),
    }


# ── Phase 3 addition ─────────────────────────────────────────────────────────

_CPI_COMPONENT_LABELS = {
    "CUSR0000SAH1":   "Shelter",
    "CUSR0000SA0E":   "Energy",
    "CUSR0000SAF11":  "Food at Home",
    "CUSR0000SACL1E": "Core Goods",
    "CUSR0000SASLE":  "Services",
}


def inflation_decomposition(fred_data: dict) -> dict:
    """
    YoY % change for 5 CPI sub-components: Shelter, Energy, Food, Core Goods, Services.
    Returns dict: { data: pd.DataFrame, target: 2.0 }
    data has one column per component; handle gracefully if cpi_components is absent.
    """
    components_raw = fred_data.get("cpi_components", {})
    if not components_raw:
        return {"data": pd.DataFrame(), "target": 2.0}

    frames = {}
    for sid, label in _CPI_COMPONENT_LABELS.items():
        df = components_raw.get(sid, pd.DataFrame())
        if df is not None and not df.empty:
            series = df.iloc[:, 0].dropna()
            yoy    = series.pct_change(12) * 100
            frames[label] = yoy

    if not frames:
        return {"data": pd.DataFrame(), "target": 2.0}

    combined = pd.DataFrame(frames).dropna(how="all")
    return {"data": combined, "target": 2.0}


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    np.random.seed(99)

    # ── Build synthetic FRED data ─────────────────────────────────────────────
    daily_idx   = pd.date_range("2000-01-01", periods=1500, freq="B")
    monthly_idx = pd.date_range("2000-01-01", periods=300,  freq="MS")
    quarterly_idx = pd.date_range("2000-01-01", periods=100, freq="QS")

    # Treasury yields
    ty_cols = list(TENOR_MAP.values())
    ty = pd.DataFrame(
        np.random.uniform(1, 6, (1500, len(ty_cols))) + np.linspace(0, -1, 1500)[:, None],
        index=daily_idx, columns=ty_cols,
    )

    # Spreads (some inversion)
    spread_vals = np.random.uniform(-0.5, 1.5, 1500)
    spreads = pd.DataFrame({"10Y-2Y": spread_vals,
                            "10Y-3M": spread_vals * 1.1}, index=daily_idx)

    # Monthly macro
    cpi_vals  = 100 * np.cumprod(1 + np.random.uniform(0.001, 0.006, 300))
    cpi_df    = pd.DataFrame({"CPI": cpi_vals, "Core CPI": cpi_vals * 0.98}, index=monthly_idx)
    pce_df    = pd.DataFrame({"PCE": cpi_vals * 0.97, "Core PCE": cpi_vals * 0.95}, index=monthly_idx)
    ff_df     = pd.DataFrame({"Fed Funds Rate": np.clip(np.random.uniform(0, 5.5, 300), 0, None)}, index=monthly_idx)
    ur_df     = pd.DataFrame({"Unemployment Rate": np.random.uniform(3.5, 8, 300)}, index=monthly_idx)
    gdp_df    = pd.DataFrame({"Real GDP": 20000 * np.cumprod(1 + np.random.uniform(0, 0.01, 100))}, index=quarterly_idx)
    ry_df     = pd.DataFrame({"Real Yield 10Y": np.random.uniform(-1, 2.5, 1500),
                               "Breakeven 10Y": np.random.uniform(1.5, 3.5, 1500)}, index=daily_idx)
    cs_df     = pd.DataFrame({"HY OAS": np.random.uniform(2, 8, 1500),
                               "IG OAS": np.random.uniform(0.5, 2.5, 1500)}, index=daily_idx)
    ff_ur_df  = pd.DataFrame({"Fed Funds Rate": ff_df["Fed Funds Rate"].reindex(monthly_idx),
                               "Unemployment Rate": ur_df["Unemployment Rate"].reindex(monthly_idx)})

    macro_series = {
        "FEDFUNDS": ff_df,
        "UNRATE":   ur_df,
    }

    fred = {
        "treasury_yields":        ty,
        "yield_curve_spreads":    spreads,
        "cpi":                    cpi_df,
        "pce":                    pce_df,
        "macro_series":           macro_series,
        "gdp":                    gdp_df,
        "real_yield_breakeven":   ry_df,
        "credit_spreads":         cs_df,
        "fed_funds_unemployment": ff_ur_df,
        "pce":                    pce_df,
    }

    print("── yield_curve_snapshot ─────────────────────────")
    snap = yield_curve_snapshot(fred)
    print(f"  shape  : {snap['shape']}")
    print(f"  summary: {snap['summary']}")
    print(f"  2Y={snap['rates']['2Y']:.3f}  10Y={snap['rates']['10Y']:.3f}")

    print("\n── treasury_yields_history ──────────────────────")
    tyh = treasury_yields_history(fred, period="1Y")
    print(f"  shape={tyh.shape}  cols={list(tyh.columns)}")

    print("\n── key_rates_snapshot ───────────────────────────")
    krs = key_rates_snapshot(fred)
    for r in krs:
        print(f"  {r['label']}: {r['value']} ({r['direction']})")

    print("\n── curve_spreads ────────────────────────────────")
    cs = curve_spreads(fred)
    print(f"  data shape={cs['data'].shape}")
    print(f"  10Y-2Y inversions: {len(cs['inversion_10y2y'])}")

    print("\n── inflation_series ─────────────────────────────")
    inf = inflation_series(fred)
    print(f"  data shape={inf['data'].shape}  cols={list(inf['data'].columns)}")
    print(f"  target={inf['target']}%")

    print("\n── gdp_growth ───────────────────────────────────")
    gdp = gdp_growth(fred)
    print(f"  rows={len(gdp['dates'])}  last_val={gdp['values'][-1]:.2f}%  last_color={gdp['colors'][-1]}")

    print("\n── macro_pulse ──────────────────────────────────")
    mp = macro_pulse(fred)
    print(f"  dates={len(mp['dates'])}  spread rows={sum(v is not None for v in mp['spread_10y2y'])}")

    print("\n── compute_all_fixed_income ─────────────────────")
    out = compute_all_fixed_income({"fred": fred})
    print(f"  keys: {list(out.keys())}")
    print("All fixed income checks passed.")
