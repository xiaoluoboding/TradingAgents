"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        options_report = state.get("options_report", "")
        options_trader_plan = state.get("options_trader_plan", "")
        options_instruction = (
            " This is an options workflow. Treat the Options Trader report as primary for contract selection, Greeks, IV, liquidity, expiration, and risk management; do not reduce an option recommendation to the underlying stock direction."
            if options_report else ""
        )
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Stock Trader. Analyze the stock using market price action, technical indicators, fundamentals, news, sentiment, macro conditions, and the overall research plan to make a buy, sell, or hold decision. "
                    "The Options Analyst and Options Trader reports are supplemental and must not impose their OPEN/WAIT rules on the stock decision. Do not let an options contract recommendation replace holistic stock and macro analysis. "
                    + grounding
                    + NO_EXTERNAL_TOOLS
                    + options_instruction
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    f"{report_section}"
                    f"Options reports (reference only; do not use their opening rules for the stock decision):\n"
                    f"Options Analyst report:\n{options_report}\n\n"
                    f"Options Trader recommendation:\n{options_trader_plan}\n\n"
                    f"Make an informed, strategic trading decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
