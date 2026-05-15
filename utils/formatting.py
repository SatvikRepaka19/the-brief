"""Color coding helpers and number formatting for the frontend JSON output."""


def pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string, e.g. 0.0342 → '3.42%'."""
    return f"{value * 100:.{decimals}f}%"


def bps(value: float) -> str:
    """Format a decimal rate as basis points string, e.g. 0.0050 → '50 bps'."""
    return f"{value * 10_000:.0f} bps"


def signed(value: float, decimals: int = 2) -> str:
    """Format with explicit sign, e.g. 1.23 → '+1.23', -0.5 → '-0.50'."""
    return f"{value:+.{decimals}f}"


def color_return(value: float) -> str:
    """Return 'green', 'red', or 'neutral' based on sign of a return."""
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "neutral"


def color_spread(value: float, wider_is_bad: bool = True) -> str:
    """Credit spreads: wider = risk-off (red). Narrower = green."""
    if wider_is_bad:
        return "red" if value > 0 else "green"
    return "green" if value > 0 else "red"


def fmt_large(value: float) -> str:
    """Human-readable large numbers: 1_200_000 → '1.2M', 34_000 → '34.0K'."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"
