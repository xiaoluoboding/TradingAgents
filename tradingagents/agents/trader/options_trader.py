"""Options Trader: converts research and Greeks into executable option strategies."""

from langchain_core.messages import AIMessage

from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state, get_language_instruction


def create_options_trader(llm):
    def options_trader_node(state, name="Options Trader"):
        instrument_context = get_instrument_context_from_state(state)
        plan = state.get("investment_plan", "")
        options_report = state.get("options_report", "")
        prompt = f"""You are the Options Trader on the Trading Team. Turn the Research Manager's directional view and the Options Analyst's live chain/Greeks report into concrete, executable options strategy recommendations.

{instrument_context}

Research Manager plan:
{plan}

Options Analyst report:
{options_report or 'No Options Analyst report was selected; state that contract data is insufficient and avoid inventing a contract.'}

The desk's first strategy is a cash-secured put (CSP), not an uncovered put: sell an approximately 45-DTE ATM or near-ATM put when the thesis is bullish/neutral, the effective purchase price is acceptable, liquidity is usable, premium compensates for measured downside risk, and cash collateral is available. Apply the 45/21 rule: take profit, buy back, or roll around 21 DTE rather than automatically holding to expiration. When those opening parameters fit, explicitly label the strategy **OPEN — CSP SELLER STRATEGY**. Assignment exposure, negative Gamma, or normal market uncertainty are management risks and must not by themselves turn a qualified setup into WAIT.

The second strategy is an **OTM SELL PUT INCOME STRATEGY** for collecting premium without intending to acquire shares. It is a single-leg OTM short put. Start with Delta 0.10-0.20 and choose a concrete expiration independently of the ATM 45-DTE rule. If a liquid contract offers meaningful premium and the strike has reasonable expected-move or technical support distance, label it **OPEN — OTM SELL PUT INCOME STRATEGY**. If 0.10-0.20 is too aggressive, move to a lower Delta and continue searching for an OPEN candidate instead of defaulting to no trade. Choose expiry from volatility, catalysts, Theta efficiency, premium quality, and liquidity. Define early profit-taking, buyback, roll, and stop triggers so assignment is actively avoided. Include exact strike, expiry, current bid/ask, recommended limit credit, minimum acceptable credit, breakeven, Greeks, margin/collateral assumption, and position sizing.

The third strategy is an optional LEAPS Call. Label it **OPEN — LEAPS CALL BUYER STRATEGY** when there is a genuine long-term bullish thesis, at least one year to expiration, a preferably deep-ITM liquid contract with Delta near/above 0.85, and acceptable cost, Theta, spread, and IV/term structure. These are practical opening parameters, not a demand for a perfect setup. Ordinary premium-at-risk is not a reason to reject an otherwise qualified LEAPS. If a material parameter is unavailable or fails, label it WAIT or NOT APPLICABLE and name that parameter.

Evaluate the three strategies independently; more than one may be OPEN. Then rank all OPEN candidates and name the best one to deploy now. The decision policy is parameter-driven and action-oriented: if the live contract data satisfies a strategy's opening parameters, recommend opening it. Do not return three WAIT/NO-TRADE decisions merely because risks exist. If none qualifies, cite the concrete failed numeric or factual parameter for every strategy. Never invent missing quotes or Greeks. Clearly distinguish market facts, account assumptions, and management rules.""" + get_language_instruction()
        response = llm.invoke(prompt)
        report = response.content
        return {
            "messages": [AIMessage(content=report)],
            "options_trader_plan": report,
            "sender": name,
        }

    return options_trader_node
