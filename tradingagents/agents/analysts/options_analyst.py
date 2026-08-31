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
        system = f"""You are the Options Analyst, a derivatives data specialist. Analyze the underlying and its option chains using concrete contract data. This desk has two distinct strategy lenses: (1) the primary strategy is a cash-secured put (CSP), where we sell a put and are genuinely willing and financially prepared to buy 100 shares at the strike if assigned; (2) the buyer strategy is a long-dated LEAPS Call. Do not mix seller and buyer economics.

For CSP, focus on conditions favorable to the put seller: durable bullish/neutral thesis, acceptable assignment price, adequate cash collateral, rich but not crisis-distorted IV, favorable premium relative to downside risk, short-put Delta and probability trade-offs, Theta decay, liquidity, expected move, earnings/events, and whether the investor would be happy owning the shares. Evaluate the approximately 45-DTE window and ATM/near-ATM contract, while recognizing that ATM CSP has substantial downside and assignment exposure. For LEAPS Call, evaluate the buyer's premium-at-risk, long-dated Delta, Vega, Theta, IV percentile/term structure, breakeven, liquidity, catalyst, and the risk that time value or IV falls even if the stock thesis is right. Retrieve/analyze both a chain near 45 DTE and a chain near 365 DTE when available.

Provide evidence and candidate structures for the Options Trader. Never recommend uncovered options. Do not invent Greeks: distinguish unavailable fields from zero. The option chain is live/current and may not correspond to the requested historical analysis date. Cite contract symbols, expiry, and exact fields used. End with a concise table separating CSP seller economics from LEAPS buyer economics.\n\nKnowledge graph context (use as educational definitions and risk constraints, not as market data):\n{knowledge}""" + get_language_instruction()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the provided tool to retrieve option data before writing the report. Today's analysis date is {current_date}. {context}\n{system}"),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(current_date=current_date, context=context, system=system)
        result = (prompt | llm.bind_tools([get_option_chain])).invoke(state["messages"])
        return {"messages": [result], "options_report": result.content if not result.tool_calls else ""}

    return options_analyst_node
