from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.utils.options_tools import get_option_chain
from tradingagents.dataflows.config import get_config
from tradingagents.knowledge import retrieve_options_knowledge


def create_options_analyst(llm):
    def options_analyst_node(state):
        current_date = state["trade_date"]
        context = get_instrument_context_from_state(state)
        knowledge = state.get("options_knowledge_context", "").strip()
        if not knowledge:
            knowledge = retrieve_options_knowledge(
                state["company_of_interest"],
                get_config().get("options_knowledge_path"),
            )
        system = f"""You are the Options Analyst, a derivatives data specialist. Analyze the underlying and its option chains using concrete contract data. This desk has three independent strategy lenses: (1) an approximately 45-DTE ATM/near-ATM cash-secured put (CSP), where assignment is acceptable and cash is reserved; (2) a dynamically dated low-Delta OTM single-leg Sell Put for premium income without intending to acquire shares; and (3) an optional long-dated LEAPS Call. Do not mix their objectives or economics.

For CSP, focus on conditions favorable to the put seller: bullish/neutral thesis, acceptable assignment price, adequate cash collateral, useful premium relative to downside risk, short-put Delta, Theta, Gamma, liquidity, expected move, earnings/events, and willingness to own the shares. Evaluate the approximately 45-DTE ATM/near-ATM contract. If these opening parameters are met, label it OPEN rather than rejecting it merely because assignment, negative Gamma, or ordinary market risk exists; those are management risks, not automatic vetoes.

Also construct a separate OTM single-leg Sell Put income candidate for an investor whose objective is collecting premium, not acquiring shares. Start with a short-put Delta of 0.10-0.20. Select the expiration from the available chain based on volatility, catalyst calendar, Theta decay, premium quality, and liquidity; do not apply the ATM CSP 45-DTE rule to this strategy. If a liquid contract in that Delta band offers meaningful premium and its strike is reasonably outside the expected move or supported by the price structure, label it OPEN. If that band is too close to the expected move, move to a lower Delta and still seek an OPEN candidate. Only reject when no liquid contract offers adequate premium for the measured risk, the thesis is bearish, a near-term event makes the position unmanageable, or margin capacity is explicitly inadequate. State the planned early buyback/roll/stop trigger and flag that this is not a share-acquisition strategy.

For LEAPS Call, evaluate the buyer's premium-at-risk, long-dated Delta, Vega, Theta, IV/term structure, breakeven, liquidity, and long-term thesis. If a liquid contract with at least one year to expiry, preferably deep ITM with Delta near/above 0.85, has acceptable cost and IV, label it OPEN. Do not demand perfect conditions or reject it for ordinary premium-at-risk alone. Retrieve/analyze chains near 45 DTE and 365 DTE, plus multiple OTM expirations such as roughly 14, 30, 60, and 90 DTE when available.

Evaluate all three strategies independently as OPEN, WAIT, or NOT APPLICABLE. Parameter fit should lead to OPEN. Soft warnings must become position-management notes rather than vetoes. If all three are not OPEN, identify a specific failed parameter for each; generic caution is insufficient.

Provide evidence and candidate structures for the Options Trader. A single-leg OTM Sell Put is allowed in this premium-income workflow, subject to explicit margin, leverage, early-exit, and assignment-risk controls. Do not invent Greeks: distinguish unavailable fields from zero. The option chain is live/current and may not correspond to the requested historical analysis date. Cite contract symbols, expiry, and exact fields used. End with a concise table covering CSP seller economics, OTM Sell Put income economics, and LEAPS buyer economics.\n\nKnowledge graph context (use as educational definitions and risk constraints, not as market data):\n{knowledge}""" + get_language_instruction()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the provided tool to retrieve option data before writing the report. Today's analysis date is {current_date}. {context}\n{system}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(current_date=current_date, context=context, system=system)
        result = (prompt | llm.bind_tools([get_option_chain])).invoke(state["messages"])
        return {"messages": [result], "options_report": result.content if not result.tool_calls else ""}

    return options_analyst_node
