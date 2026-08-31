"""Options Trader: converts research and Greeks into a defined-risk strategy recommendation."""

from langchain_core.messages import AIMessage

from tradingagents.agents.utils.agent_utils import get_instrument_context_from_state, get_language_instruction


def create_options_trader(llm):
    def options_trader_node(state, name="Options Trader"):
        instrument_context = get_instrument_context_from_state(state)
        plan = state.get("investment_plan", "")
        options_report = state.get("options_report", "")
        prompt = f"""You are the Options Trader on the Trading Team. Turn the Research Manager's directional view and the Options Analyst's live chain/Greeks report into a concrete, risk-defined options strategy recommendation.

{instrument_context}

Research Manager plan:
{plan}

Options Analyst report:
{options_report or 'No Options Analyst report was selected; state that contract data is insufficient and avoid inventing a contract.'}

The desk's primary income strategy is a cash-secured put (CSP), not an uncovered/naked put: sell an approximately 45-DTE ATM or near-ATM put only when the account has cash to take assignment and the investor genuinely wants to own 100 shares at the strike. Apply the 45/21 rule: take profit or buy back around 21 DTE rather than automatically holding to expiration. This is a hypothesis, not a mandatory recommendation. Explicitly answer whether a 45-day ATM CSP should be the PRIMARY strategy for this underlying today, with emphasis on conditions favorable to the seller: durable bullish/neutral thesis, acceptable effective purchase price, rich-but-not-panic IV, premium versus downside gap risk, short-put Delta, Theta, Gamma, liquidity, earnings/events, collateral, and willingness to own the stock. Do not treat a high premium alone as favorable.

Also provide an **OTM NO-SHARES INCOME STRATEGY** for the separate objective of collecting premium without taking stock. Use a defined-risk OTM put credit spread: sell an OTM put with Delta 0.10-0.20 when that band is stable, and buy a farther OTM protective put. Judge stability using expected move, IV/skew, Gamma, earnings/catalysts, liquidity, spread width, credit, max loss, breakeven, and return on defined risk. If the underlying is not stable in the 0.10-0.20 range, recommend a lower-Delta short leg and explain why. Never use an uncovered short put for this objective. Require closing the spread before expiration; do not promise that any Delta makes assignment risk zero. Include exact strikes, expiry, credit/debit, max profit, max loss, breakeven, Greeks, liquidity, exit trigger, and position sizing.

LEAPS Call is OPTIONAL and must not be presented as a default alternative or as a replacement for CSP. Include a LEAPS Call recommendation only if the knowledge-graph conditions are satisfied by the current data: a genuine long-term investment thesis (not short-term speculation), at least one year to expiration, preferably deep ITM rather than ATM/OTM, Delta generally near or above 0.85, manageable Theta/cost, and sufficiently liquid contracts with acceptable bid-ask spread and IV/term-structure pricing. If any material condition is not met or the data is unavailable, omit the LEAPS strategy from the recommended-strategies section and state briefly that LEAPS conditions were not met. Keep CSP seller economics and any qualifying LEAPS buyer economics clearly separated.

After deciding whether CSP is suitable, evaluate the knowledge-graph strategies supported by the available chain and select the single best non-CSP strategy only when its own entry conditions and risk profile are supported by current data. The OTM no-shares spread is eligible when stable premium collection and avoiding shares is the objective. Do not force an alternative: if no strategy qualifies, say so. LEAPS is one optional candidate, not the default alternative.

Provide one primary strategy and at most one qualifying alternative. The primary recommendation must explicitly be **CSP SELLER STRATEGY** or **NEITHER**. Only when all LEAPS conditions above are met may you add **LEAPS CALL BUYER STRATEGY**; otherwise do not mention it as a recommendation. Name exact contract(s), expiry, strike(s), credit/debit, max profit, max loss, breakeven/effective purchase price, Delta/Theta/IV rationale, liquidity checks, catalyst, the 21-DTE CSP profit-taking/buyback/roll rule, exit plan, assignment/collateral requirements, invalidation, and position sizing. Never invent missing quotes or Greeks, and do not recommend naked short options. Clearly distinguish market facts from assumptions.""" + get_language_instruction()
        response = llm.invoke(prompt)
        report = response.content
        return {
            "messages": [AIMessage(content=report)],
            "options_trader_plan": report,
            "sender": name,
        }

    return options_trader_node
