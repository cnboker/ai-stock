import pandas as pd
from strategy.equity_policy import TradeIntent

"""
典型触发场景:decision.force_reduce
场景	触发条件	解释
止损	股票价格跌破某个止损价	系统发现当前持仓亏损过大，需要强制减仓，忽略模型动作
风险控制	当日总亏损/浮动亏损超过阈值	风控策略限制账户整体风险，所有触发仓位可能被强制减仓
仓位上限	当前持仓超过策略允许最大仓位	强制减仓以回到安全范围
异常事件	股票停牌、数据异常、行情异常	系统检测到异常，立刻减仓或清仓
策略切换	新的市场 regime 切换，当前策略不适用	强制降低仓位或平掉不安全的仓位
"""


def execute_equity_action(
    *,
    decision: TradeIntent,
    position_mgr,
    ticker: str,
    last_price: float,
):
    """
    唯一允许修改仓位的入口
    返回: dict, 用于动态表格显示
    """
    # 默认 action 为 HOLD
    final_action = "HOLD"
    pos_dict = position_mgr.pos_to_dict(ticker=ticker)
    if decision.action == "LIQUIDATE":
        position_mgr.close(
            ticker=ticker,
            price=last_price,
            reason=decision.reason,
        )
        final_action = "LIQUIDATE"

    # 含义：强制减仓，不管当前信号如何。

    elif decision.force_reduce:
        position_mgr.reduce(
            ticker=ticker,
            strength=decision.reduce_strength,
            price=last_price,
            reason=decision.reason,
        )
        final_action = "REDUCE"

    # 非确认信号 → 不交易
    elif not decision.confirmed:
        final_action = "HOLD"

    # 执行动作
    else:
        if decision.action == "LONG":
            position_mgr.open_or_add(
                ticker=ticker,
                strength=decision.confidence * decision.gate_mult,
                price=last_price,
                reason=decision.reason,
            )
            final_action = "LONG"

        elif decision.action == "REDUCE":
            position_mgr.reduce(
                ticker=ticker,
                strength=decision.reduce_strength,
                price=last_price,
                reason=decision.reason,
            )
            final_action = "REDUCE"

        elif decision.action == "SHORT":
            position_mgr.open_short(
                ticker=ticker,
                strength=decision.confidence * decision.gate_mult,
                price=last_price,
                reason=decision.reason,
            )
            final_action = "SHORT"

    # 统一返回字典
    return {
        "ticker": ticker,
        **pos_dict,
        **decision.__dict__,
        "action": final_action,
    }
