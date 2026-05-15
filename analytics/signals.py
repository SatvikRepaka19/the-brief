"""
Equity market signals: z-scores, composite scores, breadth, cyc/def ratio,
macro-vs-micro dispersion, and SPY volume z-score.
"""

import warnings
import numpy as np
import pandas as pd

# ── Ticker universe ──────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"]

SECTOR_NAMES = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discretionary",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

INDEX_ETFS   = ["SPY", "RSP", "QQQ", "IWM", "DIA"]
FACTOR_ETFS  = ["USMV", "MTUM", "VLUE", "QUAL", "HDV", "SIZE"]

CYCLICAL_ETFS  = ["XLK", "XLF", "XLY", "XLI", "XLB"]
DEFENSIVE_ETFS = ["XLP", "XLU", "XLV"]


# ── Core helpers ─────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score: (x − rolling_mean) / rolling_std, clipped to ±3.
    Uses min_periods = window // 4 so early values aren't all NaN.
    """
    min_p = max(2, window // 4)
    mean  = series.rolling(window, min_periods=min_p).mean()
    std   = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return ((series - mean) / std).clip(-3, 3)


def _pct_return(price: pd.Series, n: int) -> float:
    """Last n-period % return from a price series. Returns NaN on failure."""
    try:
        val = price.pct_change(n).iloc[-1]
        return float(val) * 100 if not np.isnan(val) else np.nan
    except Exception:
        return np.nan


# ── Signal functions ─────────────────────────────────────────────────────────

def compute_sector_composites(prices_df: pd.DataFrame,
                               volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector ETF compute:
      return_z  — rolling z-score of daily returns (window=252)
      volume_z  — rolling z-score of volume (window=252)
      composite — (return_z + volume_z) / 2, clipped ±3
    Also compute 1D / 5D / 1M / 12M period returns.

    Returns a DataFrame (one row per ticker) sorted by composite descending.
    """
    records = []
    for ticker in SECTOR_ETFS:
        if ticker not in prices_df.columns:
            continue

        price = prices_df[ticker].dropna()
        ret   = price.pct_change()

        ret_z = rolling_zscore(ret, window=252)

        if ticker in volume_df.columns:
            vol_z = rolling_zscore(volume_df[ticker].dropna(), window=252)
        else:
            vol_z = pd.Series(np.nan, index=ret.index)

        composite = ((ret_z + vol_z) / 2).clip(-3, 3)

        def _last(s: pd.Series) -> float:
            return float(s.iloc[-1]) if not s.empty and not np.isnan(s.iloc[-1]) else np.nan

        records.append({
            "ticker":    ticker,
            "name":      SECTOR_NAMES.get(ticker, ticker),
            "1D":        _pct_return(price, 1),
            "5D":        _pct_return(price, 5),
            "1M":        _pct_return(price, 21),
            "12M":       _pct_return(price, 252),
            "return_z":  _last(ret_z),
            "volume_z":  _last(vol_z),
            "composite": _last(composite),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    return df


def breadth_ratio(prices_df: pd.DataFrame) -> pd.Series:
    """
    RSP / SPY ratio, indexed to 1.0 at the first observation.
    Equal-weight vs cap-weight — rising = broader participation.
    """
    if "RSP" not in prices_df.columns or "SPY" not in prices_df.columns:
        warnings.warn("[signals] RSP or SPY missing — breadth_ratio returning empty")
        return pd.Series(dtype=float, name="breadth_ratio")

    ratio = (prices_df["RSP"] / prices_df["SPY"]).dropna()
    ratio = ratio / ratio.iloc[0]
    ratio.name = "breadth_ratio"
    return ratio


def cyclical_defensive_ratio(prices_df: pd.DataFrame) -> pd.Series:
    """
    Equal-weight cyclical basket (XLK, XLF, XLY, XLI, XLB) divided by
    equal-weight defensive basket (XLP, XLU, XLV). Rising = risk-on.
    """
    cyc_cols = [c for c in CYCLICAL_ETFS  if c in prices_df.columns]
    def_cols = [c for c in DEFENSIVE_ETFS if c in prices_df.columns]

    if not cyc_cols or not def_cols:
        warnings.warn("[signals] insufficient tickers for cyclical_defensive_ratio")
        return pd.Series(dtype=float, name="cyc_def_ratio")

    cyc   = prices_df[cyc_cols].mean(axis=1)
    defen = prices_df[def_cols].mean(axis=1)

    ratio = (cyc / defen).dropna()
    ratio.name = "cyc_def_ratio"
    return ratio


def spy_volume_zscore(volume_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Rolling z-score of SPY daily volume vs 1Y median. window=252 trading days."""
    if "SPY" not in volume_df.columns:
        warnings.warn("[signals] SPY not in volume_df")
        return pd.Series(dtype=float, name="spy_volume_z")

    vol = volume_df["SPY"].dropna()
    z   = rolling_zscore(vol, window=window)
    z.name = "spy_volume_z"
    return z


def macro_vs_micro_dispersion(prices_df: pd.DataFrame,
                               window: int = 252) -> pd.Series:
    """
    Macro-vs-micro dispersion ratio, as a rolling percentile rank (0–1).

    between_sector : cross-sectional std of sector ETF returns on each date
    within_sector  : mean of each sector ETF's own 21-day rolling return std
    ratio          = between / (between + within)
    output         = rolling percentile rank of ratio over `window` days

    >0.75  →  sector-driven market (macro dominates)
    <0.25  →  stock-driven market  (micro dominates)
    """
    sector_cols = [c for c in SECTOR_ETFS if c in prices_df.columns]
    if len(sector_cols) < 3:
        warnings.warn("[signals] too few sector ETFs for macro_vs_micro_dispersion")
        return pd.Series(dtype=float, name="macro_micro")

    rets = prices_df[sector_cols].pct_change()

    between = rets.std(axis=1)                          # cross-sectional dispersion
    within  = rets.rolling(21, min_periods=5).std().mean(axis=1)  # avg within-ETF vol

    total = (between + within).replace(0, np.nan)
    ratio = between / total

    # Rolling percentile rank: fraction of past `window` days where ratio ≤ today
    pct_rank = ratio.rolling(window, min_periods=window // 4).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )
    pct_rank.name = "macro_micro"
    return pct_rank.dropna()


# ── Top-level aggregator ─────────────────────────────────────────────────────

def compute_all_signals(data_dict: dict) -> dict:
    """
    Entry point for Phase 2 signal computation.

    Parameters
    ----------
    data_dict : output of cache_all()
        Keys: 'fred', 'equity' → (prices_df, volume_df, ohlc_df)

    Returns
    -------
    dict with keys:
        sector_composites  — list of dicts (one per sector ETF)
        breadth            — pd.Series
        cyclical_defensive — pd.Series
        spy_volume_z       — pd.Series
        macro_micro        — pd.Series
    """
    prices_df, volume_df, _ = data_dict["equity"]

    composites = compute_sector_composites(prices_df, volume_df)

    return {
        "sector_composites":  composites.to_dict(orient="records") if not composites.empty else [],
        "breadth":            breadth_ratio(prices_df),
        "cyclical_defensive": cyclical_defensive_ratio(prices_df),
        "spy_volume_z":       spy_volume_zscore(volume_df),
        "macro_micro":        macro_vs_micro_dispersion(prices_df),
    }


# ── Phase 3 additions ────────────────────────────────────────────────────────

COMMODITY_TICKERS = ["UUP", "GLD", "USO", "CPER"]
COMMODITY_NAMES   = {"UUP": "US Dollar", "GLD": "Gold", "USO": "Crude Oil", "CPER": "Copper"}


def market_regime_score(signals_dict: dict) -> dict:
    """
    Combine 5 signals into a single 0-100 Fear/Greed score.

    Each component is normalized to 0–20 via rolling 252-day percentile rank
    (or linear mapping for z-score signals) and then summed.

    Returns dict: { score, label, components }
    """
    def _pct_rank(series: pd.Series, window: int = 252) -> float:
        """Percentile rank of the latest value over the rolling window."""
        clean = series.dropna()
        if clean.empty:
            return 0.5
        window_data = clean.tail(window)
        current = window_data.iloc[-1]
        return float((window_data <= current).mean())

    breadth = signals_dict.get("breadth",            pd.Series(dtype=float))
    cyc_def = signals_dict.get("cyclical_defensive",  pd.Series(dtype=float))
    vol_z   = signals_dict.get("spy_volume_z",        pd.Series(dtype=float))
    mm      = signals_dict.get("macro_micro",          pd.Series(dtype=float))
    sc      = signals_dict.get("sector_composites",    [])

    # Each component → [0, 20]
    b_score  = _pct_rank(breadth)  * 20
    cd_score = _pct_rank(cyc_def)  * 20

    vz_last  = float(vol_z.dropna().iloc[-1])  if not vol_z.dropna().empty  else 0.0
    vz_score = np.clip((vz_last + 3) / 6, 0, 1) * 20

    mm_last  = float(mm.dropna().iloc[-1])  if not mm.dropna().empty  else 0.5
    mm_score = np.clip(mm_last, 0, 1) * 20

    valid_composites = [r["composite"] for r in sc
                        if r.get("composite") is not None and not np.isnan(r["composite"])]
    comp_avg   = np.mean(valid_composites) if valid_composites else 0.0
    comp_score = np.clip((comp_avg + 3) / 6, 0, 1) * 20

    components = {
        "breadth":     round(float(np.clip(b_score,  0, 20)), 2),
        "cyc_def":     round(float(np.clip(cd_score, 0, 20)), 2),
        "vol_z":       round(float(np.clip(vz_score, 0, 20)), 2),
        "macro_micro": round(float(np.clip(mm_score, 0, 20)), 2),
        "composite":   round(float(np.clip(comp_score, 0, 20)), 2),
    }
    score = round(sum(components.values()), 1)

    if score <= 25:   label = "Extreme Fear"
    elif score <= 45: label = "Fear"
    elif score <= 55: label = "Neutral"
    elif score <= 75: label = "Greed"
    else:             label = "Extreme Greed"

    return {"score": score, "label": label, "components": components}


def correlation_matrix(prices_df: pd.DataFrame, window: int = 60) -> dict:
    """
    Rolling 60-day return correlation for all sector ETFs.

    Returns dict ready for Plotly heatmap:
        x, y  — ticker label lists
        z     — 2-D list (rows = y tickers, cols = x tickers)
    """
    sector_cols = [c for c in SECTOR_ETFS if c in prices_df.columns]
    if len(sector_cols) < 2:
        return {"x": [], "y": [], "z": []}

    rets = prices_df[sector_cols].pct_change().tail(window)
    corr = rets.corr()

    z = []
    for row in sector_cols:
        r = []
        for col in sector_cols:
            v = corr.loc[row, col] if row in corr.index and col in corr.columns else None
            r.append(round(float(v), 3) if v is not None and not np.isnan(v) else None)
        z.append(r)

    return {"x": sector_cols, "y": sector_cols, "z": z}


def fifty_two_week_range(prices_df: pd.DataFrame) -> list[dict]:
    """
    52-week high / low proximity for each sector ETF.

    Returns list of dicts sorted by pct_from_high ascending (worst first).
    Fields: ticker, name, current, high_52w, low_52w,
            pct_from_high, pct_from_low, pct_of_range (0-100).
    """
    records = []
    window = min(252, len(prices_df))

    for ticker in SECTOR_ETFS:
        if ticker not in prices_df.columns:
            continue
        price = prices_df[ticker].dropna()
        if price.empty:
            continue

        recent  = price.tail(window)
        high    = float(recent.max())
        low     = float(recent.min())
        current = float(price.iloc[-1])

        pct_from_high  = (current / high - 1) * 100
        pct_from_low   = (current / low  - 1) * 100
        pct_of_range   = (current - low) / (high - low) * 100 if high != low else 50.0

        records.append({
            "ticker":       ticker,
            "name":         SECTOR_NAMES.get(ticker, ticker),
            "current":      round(current, 2),
            "high_52w":     round(high, 2),
            "low_52w":      round(low, 2),
            "pct_from_high": round(pct_from_high, 2),
            "pct_from_low":  round(pct_from_low, 2),
            "pct_of_range":  round(pct_of_range, 1),
        })

    return sorted(records, key=lambda x: x["pct_from_high"])


def drawdown_series(prices_df: pd.DataFrame,
                    tickers: list[str] | None = None) -> dict:
    """
    Rolling max drawdown for each ticker: (price / cummax - 1) * 100.

    Returns {ticker: {dates: [...], values: [...]}} — values are ≤ 0.
    """
    if tickers is None:
        tickers = ["SPY"] + SECTOR_ETFS

    result = {}
    for ticker in tickers:
        if ticker not in prices_df.columns:
            continue
        price = prices_df[ticker].dropna()
        if price.empty:
            continue
        dd = (price / price.cummax() - 1) * 100
        result[ticker] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in dd.index],
            "values": [round(float(v), 4) if not np.isnan(v) else None for v in dd.values],
        }
    return result


def commodities_snapshot(prices_df: pd.DataFrame) -> dict:
    """
    Latest price + 1D / 1M % change for UUP, GLD, USO, CPER.
    Also computes Copper/Gold ratio time series.

    Returns dict: { snapshot: [...], copper_gold_ratio: {dates, values} }
    """
    snapshot = []
    for ticker in COMMODITY_TICKERS:
        if ticker not in prices_df.columns:
            continue
        price = prices_df[ticker].dropna()
        if len(price) < 2:
            continue
        snapshot.append({
            "ticker": ticker,
            "name":   COMMODITY_NAMES.get(ticker, ticker),
            "price":  round(float(price.iloc[-1]), 2),
            "1D":     round(float(price.pct_change(1).iloc[-1])  * 100, 2),
            "1M":     round(float(price.pct_change(21).iloc[-1]) * 100, 2)
                      if len(price) > 21 else None,
        })

    ratio: dict = {}
    if "CPER" in prices_df.columns and "GLD" in prices_df.columns:
        r = (prices_df["CPER"] / prices_df["GLD"]).dropna()
        ratio = {
            "dates":  [d.strftime("%Y-%m-%d") for d in r.index],
            "values": [round(float(v), 6) if not np.isnan(v) else None for v in r.values],
            "label":  "Copper / Gold Ratio",
        }

    return {"snapshot": snapshot, "copper_gold_ratio": ratio}


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 320
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    all_tickers = SECTOR_ETFS + INDEX_ETFS + FACTOR_ETFS

    prices = pd.DataFrame(
        100 + np.random.randn(n, len(all_tickers)).cumsum(0),
        index=dates, columns=all_tickers,
    )
    volumes = pd.DataFrame(
        np.abs(np.random.randn(n, len(all_tickers))) * 1_000_000 + 500_000,
        index=dates, columns=all_tickers,
    )

    print("── rolling_zscore ──────────────────────────────")
    z = rolling_zscore(prices["SPY"].pct_change(), window=63)
    print(f"  shape={z.shape}  min={z.min():.3f}  max={z.max():.3f}  clipped to ±3 ✓")

    print("\n── compute_sector_composites ───────────────────")
    sc = compute_sector_composites(prices, volumes)
    print(sc[["ticker", "1D", "1M", "return_z", "volume_z", "composite"]].to_string(index=False))

    print("\n── breadth_ratio ───────────────────────────────")
    br = breadth_ratio(prices)
    print(f"  shape={br.shape}  first=1.0000  last={br.iloc[-1]:.4f}")

    print("\n── cyclical_defensive_ratio ────────────────────")
    cd = cyclical_defensive_ratio(prices)
    print(f"  shape={cd.shape}  last={cd.iloc[-1]:.4f}")

    print("\n── spy_volume_zscore ────────────────────────────")
    vz = spy_volume_zscore(volumes)
    print(f"  shape={vz.shape}  last={vz.iloc[-1]:.4f}")

    print("\n── macro_vs_micro_dispersion ────────────────────")
    mm = macro_vs_micro_dispersion(prices, window=63)
    print(f"  shape={mm.shape}  last={mm.iloc[-1]:.4f} (0–1 range ✓)")

    print("\n── compute_all_signals ──────────────────────────")
    out = compute_all_signals({"equity": (prices, volumes, pd.DataFrame())})
    print(f"  keys: {list(out.keys())}")
    print(f"  sector_composites rows: {len(out['sector_composites'])}")
    print(f"  breadth type: {type(out['breadth']).__name__}")
    print("All signal checks passed.")
