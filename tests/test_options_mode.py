from unittest.mock import patch

import pandas as pd

from cli.models import AnalystType
from cli.utils import ANALYST_ORDER, filter_analysts_for_asset_type
from tradingagents.dataflows.options import format_option_chain
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.knowledge.options_knowledge import retrieve_options_knowledge


def test_options_analyst_is_available_for_stock_runs():
    assert AnalystType.OPTIONS in [value for _, value in ANALYST_ORDER]
    assert "options" in ANALYST_NODE_SPECS
    assert ANALYST_NODE_SPECS["options"].agent_node == "Options Analyst"


def test_option_chain_summary_preserves_greeks_and_marks_source_date():
    calls = pd.DataFrame([
        {"contractSymbol": "AAPL260918C00200000", "strike": 200, "lastPrice": 4.2,
         "bid": 4.1, "ask": 4.3, "volume": 100, "openInterest": 500,
         "impliedVolatility": .25, "delta": .62, "gamma": .03, "theta": -.08,
         "vega": .11, "rho": .02}
    ])
    output = format_option_chain(
        "AAPL", "2026-09-18", calls, "2026-08-31", underlying_price=201.25
    )
    assert "AAPL260918C00200000" in output
    assert "Delta" in output and "0.62" in output
    assert "Theta" in output and "-0.08" in output
    assert "Retrieved on: 2026-08-31" in output
    assert "Underlying price: 201.2500" in output


def test_obsidian_knowledge_retriever_prefers_ticker_and_greek_notes(tmp_path):
    (tmp_path / "AAPL.md").write_text("# AAPL\nApple options catalyst notes", encoding="utf-8")
    (tmp_path / "Delta.md").write_text("# Delta\nDirectional sensitivity", encoding="utf-8")
    (tmp_path / "unrelated.md").write_text("Cooking recipe", encoding="utf-8")
    result = retrieve_options_knowledge("AAPL", tmp_path, max_documents=2)
    assert "AAPL.md" in result
    assert "unrelated.md" not in result


@patch("tradingagents.dataflows.options.yf.Ticker")
def test_option_chain_returns_clear_sentinel_when_expirations_are_missing(ticker):
    ticker.return_value.options = ()
    from tradingagents.dataflows.options import get_option_chain_online

    assert "unavailable" in get_option_chain_online("AAPL").lower()


def test_options_trader_prompt_contains_csp_and_leaps_lenses():
    from tradingagents.agents.trader.options_trader import create_options_trader

    class FakeResponse:
        content = "strategy"

    class FakeLLM:
        def invoke(self, prompt):
            assert "cash-secured put (CSP)" in prompt
            assert "45/21" in prompt
            assert "LEAPS Call" in prompt
            assert "OTM SELL PUT INCOME STRATEGY" in prompt
            assert "0.10-0.20" in prompt
            return FakeResponse()

    node = create_options_trader(FakeLLM())
    result = node({"company_of_interest": "AAPL", "investment_plan": "bullish", "options_report": "chain"})
    assert result["options_trader_plan"] == "strategy"


def test_options_trader_prompt_opens_strategies_when_parameters_fit():
    from tradingagents.agents.trader.options_trader import create_options_trader

    class FakeResponse:
        content = "strategy"

    class FakeLLM:
        def invoke(self, prompt):
            assert "OPTIONAL" in prompt
            assert "replacement for CSP" in prompt
            assert "OPEN — CSP SELLER STRATEGY" in prompt
            assert "OPEN — OTM SELL PUT INCOME STRATEGY" in prompt
            assert "OPEN — LEAPS CALL BUYER STRATEGY" in prompt
            assert "single-leg OTM short put" in prompt
            assert "more than one may be OPEN" in prompt
            assert "normal market uncertainty" in prompt
            return FakeResponse()

    node = create_options_trader(FakeLLM())
    node({"company_of_interest": "AAPL", "investment_plan": "neutral", "options_report": "chain"})


def test_stock_trader_keeps_holistic_stock_decision_separate():
    from tradingagents.agents.trader.trader import create_trader
    from unittest.mock import patch

    class FakeLLM:
        def with_structured_output(self, *args, **kwargs):
            return self

    node = create_trader(FakeLLM())
    captured = {}

    def capture(*args):
        captured["messages"] = args[2]
        return "stock plan"

    with patch("tradingagents.agents.trader.trader.invoke_structured_or_freetext", capture):
        node({
            "company_of_interest": "AAPL",
            "instrument_context": "AAPL is Apple Inc.",
            "investment_plan": "bullish",
            "market_report": "support 200, ATR 4",
            "options_report": "OPEN OTM Sell Put",
            "options_trader_plan": "OPEN OTM SELL PUT INCOME STRATEGY",
        })
    prompt = "\n".join(message["content"] for message in captured["messages"])
    assert "You are the Stock Trader" in prompt
    assert "macro conditions" in prompt
    assert "must not impose their OPEN/WAIT rules" in prompt
