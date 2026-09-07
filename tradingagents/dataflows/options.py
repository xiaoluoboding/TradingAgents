"""Deterministic option-chain retrieval and presentation helpers."""

from datetime import datetime
from math import erf, exp, log, pi, sqrt

import pandas as pd
import yfinance as yf

from .symbol_utils import normalize_symbol


OPTION_COLUMNS = [
    "contractSymbol", "optionType", "strike", "lastPrice", "bid", "ask", "volume",
    "openInterest", "impliedVolatility", "delta", "gamma", "theta", "vega", "rho",
]


def format_option_chain(
    symbol: str,
    expiration: str,
    frame: pd.DataFrame,
    retrieved_on: str,
    underlying_price: float | None = None,
) -> str:
    """Return a compact, model-readable option chain with Greeks when supplied."""
    if frame.empty:
        return f"# Option chain unavailable for {symbol} ({expiration}): no contracts returned."
    columns = [column for column in OPTION_COLUMNS if column in frame.columns]
    data = frame[columns].copy()
    for column in columns:
        if column not in {"contractSymbol"}:
            data[column] = pd.to_numeric(data[column], errors="coerce").round(4)
    spot_line = (
        f"# Underlying price: {underlying_price:.4f}\n"
        if underlying_price is not None
        else "# Underlying price: unavailable\n"
    )
    return (
        f"# Option chain for {symbol}, expiration {expiration}\n"
        f"# Retrieved on: {retrieved_on}\n"
        + spot_line
        + f"# Contracts: {len(data)}\n"
        "# Greeks are exchange/vendor values when present; otherwise Black-Scholes estimates using the retrieved underlying price, IV, and 4% risk-free rate.\n\n"
        "# Fields: contractSymbol, optionType, strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility, Delta, Gamma, Theta, Vega, Rho\n\n"
        + data.to_csv(index=False)
    )


def get_option_chain_online(symbol: str, expiration: str | None = None, max_contracts: int = 80, target_dte: int = 45) -> str:
    """Fetch the nearest useful Yahoo option chain, including available Greeks.

    Yahoo generally exposes the live chain only; the retrieval timestamp is
    explicit so historical stock analysis cannot be mistaken for historical
    option quotes.
    """
    canonical = normalize_symbol(symbol)
    ticker = yf.Ticker(canonical)
    try:
        expirations = tuple(ticker.options or ())
    except Exception as exc:  # noqa: BLE001 - option availability is best effort
        return f"# Option chain unavailable for {canonical}: {exc}"
    if not expirations:
        return f"# Option chain unavailable for {canonical}: Yahoo returned no expirations."
    if expiration:
        selected = expiration
    else:
        today = datetime.now().date()
        selected = min(
            expirations,
            key=lambda value: abs((datetime.strptime(value, "%Y-%m-%d").date() - today).days - int(target_dte)),
        )
    if selected not in expirations:
        return f"# Option chain unavailable for {canonical}: expiration {selected} is not listed."
    try:
        chain = ticker.option_chain(selected)
    except Exception as exc:  # noqa: BLE001 - preserve a useful agent-visible sentinel
        return f"# Option chain unavailable for {canonical} ({selected}): {exc}"
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["optionType"] = "call"
    puts["optionType"] = "put"
    data = pd.concat([calls, puts], ignore_index=True)
    spot = _latest_spot(ticker)
    data = _fill_missing_greeks(data, ticker, selected, spot=spot)
    if "openInterest" in data.columns:
        data = data.sort_values("openInterest", ascending=False, na_position="last")
    return format_option_chain(
        canonical,
        selected,
        data.head(max(1, int(max_contracts))),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        underlying_price=spot,
    )


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2 * pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + erf(value / sqrt(2)))


def _latest_spot(ticker) -> float | None:
    try:
        history = ticker.history(period="1d")
        return float(history["Close"].dropna().iloc[-1])
    except Exception:
        return None


def _fill_missing_greeks(
    data: pd.DataFrame,
    ticker,
    expiration: str,
    spot: float | None = None,
) -> pd.DataFrame:
    """Estimate missing Greeks without replacing vendor-provided values."""
    required = {"strike", "impliedVolatility", "optionType"}
    if not required.issubset(data.columns):
        return data
    if spot is None:
        spot = _latest_spot(ticker)
    if spot is None:
        return data
    expiry_days = max((datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).total_seconds() / 86400, 1 / 365)
    time = expiry_days / 365
    rate = 0.04
    for index, row in data.iterrows():
        try:
            strike = float(row["strike"])
            volatility = float(row["impliedVolatility"])
            if not (spot > 0 and strike > 0 and volatility > 0):
                continue
            d1 = (log(spot / strike) + (rate + volatility * volatility / 2) * time) / (volatility * sqrt(time))
            d2 = d1 - volatility * sqrt(time)
            is_call = str(row["optionType"]).lower() == "call"
            sign = 1 if is_call else -1
            values = {
                "delta": sign * _normal_cdf(sign * d1),
                "gamma": _normal_pdf(d1) / (spot * volatility * sqrt(time)),
                "theta": (-(spot * _normal_pdf(d1) * volatility) / (2 * sqrt(time)) - sign * rate * strike * exp(-rate * time) * _normal_cdf(sign * d2)) / 365,
                "vega": spot * _normal_pdf(d1) * sqrt(time) / 100,
                "rho": sign * strike * time * exp(-rate * time) * _normal_cdf(sign * d2) / 100,
            }
            for name, value in values.items():
                if name not in data.columns or pd.isna(row.get(name)):
                    data.loc[index, name] = value
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return data
