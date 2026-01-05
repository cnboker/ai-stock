from strategy.equity_decide import EquityDecision


def execute_equity_decision(
    *,
    decision: EquityDecision,
    position_mgr,
    ticker: str,
    last_price: float,
):
    """
    唯一允许修改仓位的入口
    """

    # ===== 1. 强制减仓优先 =====
    if decision.force_reduce:
        position_mgr.reduce(
            ticker=ticker,
            strength=decision.reduce_strength,
            price=last_price,
            reason=decision.reason,
        )
        return

    # ===== 2. 非确认信号 → 不交易 =====
    if not decision.confirmed:
        return
    if decision.action == "LIQUIDATE":
        position_mgr.close(
            ticker=ticker,
            price=last_price,
            reason=decision.reason,
        )
        return
    # ===== 3. 执行动作 =====
    if decision.action == "LONG":
        position_mgr.open_or_add(
            ticker=ticker,
            strength=decision.confidence * decision.gate_mult,
            price=last_price,
            reason=decision.reason,
        )

    elif decision.action == "REDUCE":
        position_mgr.reduce(
            ticker=ticker,
            strength=decision.reduce_strength,
            price=last_price,
            reason=decision.reason,
        )

    elif decision.action == "SHORT":
        position_mgr.open_short(
            ticker=ticker,
            strength=decision.confidence * decision.gate_mult,
            price=last_price,
            reason=decision.reason,
        )
