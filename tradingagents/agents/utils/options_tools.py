from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.options import get_option_chain_online


@tool
def get_option_chain(
    symbol: Annotated[str, "underlying ticker symbol, e.g. AAPL or SPY"],
    expiration: Annotated[str | None, "optional expiration in YYYY-MM-DD; omit for nearest listed expiration"] = None,
    max_contracts: Annotated[int, "maximum number of contracts to return"] = 80,
    target_dte: Annotated[int, "target days to expiration when expiration is omitted; defaults to the 45-day strategy window"] = 45,
) -> str:
    """Retrieve current option contracts, liquidity, implied volatility, and Greeks."""
    return get_option_chain_online(symbol, expiration, max_contracts, target_dte)
