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
    output = format_option_chain("AAPL", "2026-09-18", calls, "2026-08-31")
    assert "AAPL260918C00200000" in output
    assert "Delta" in output and "0.62" in output
    assert "Theta" in output and "-0.08" in output
    assert "Retrieved on: 2026-08-31" in output


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
            return FakeResponse()

    node = create_options_trader(FakeLLM())
    result = node({"company_of_interest": "AAPL", "investment_plan": "bullish", "options_report": "chain"})
    assert result["options_trader_plan"] == "strategy"


def test_options_trader_prompt_makes_leaps_conditional():
    from tradingagents.agents.trader.options_trader import create_options_trader

    class FakeResponse:
        content = "strategy"

    class FakeLLM:
        def invoke(self, prompt):
            assert "OPTIONAL" in prompt
            assert "replacement for CSP" in prompt
            assert "omit the LEAPS strategy" in prompt
            assert "best non-CSP strategy" in prompt
            return FakeResponse()

    node = create_options_trader(FakeLLM())
    node({"company_of_interest": "AAPL", "investment_plan": "neutral", "options_report": "chain"})
