import pandas as pd
from log import signal_log
from risk.risk_manager import TradePlan
from strategy.trade_intent import TradeIntent

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
    plan: TradePlan = None,
):
    """
    唯一允许修改仓位的入口
    plan: RiskManager.evaluate() 结果，用于控制仓位/止损/止盈
    返回 dict, 用于动态表格显示
    """
   # signal_log(f"CHECK: Available Cash={position_mgr.available_cash}, Order Value={last_price * plan.size }")
    final_action = "HOLD"

    # ====================
    # 1️⃣ 强制清仓
    # ====================
    if decision.action == "LIQUIDATE":
        position_mgr.close(
            ticker=ticker,
            price=last_price,
            reason=decision.reason,
        )
        final_action = "LIQUIDATE"

    # ====================
    # 2️⃣ 强制减仓
    # ====================
    elif decision.force_reduce:
        reduce_strength = decision.reduce_strength or 0.3
        if plan is not None:
            # plan.size 可用于控制减仓数量（可按比例或者直接覆盖）
            reduce_size = plan.size if plan.size > 0 else reduce_strength
        else:
            reduce_size = reduce_strength

        position_mgr.reduce(
            ticker=ticker,
            strength=reduce_size,
            price=last_price,
            reason=decision.reason,
        )
        final_action = "REDUCE"

    # ====================
    # 3️⃣ 非确认信号 → 不交易
    # ====================
    elif not decision.confirmed:
        final_action = "HOLD"

    # ====================
    # 4️⃣ 正常交易
    # ====================
    else:
        if decision.action == "LONG" and plan and plan.allow_trade:
            if plan.size > 0:
                position_mgr.open_or_add(
                    ticker=ticker,
                    size=plan.size,
                    stop_loss=plan.stop_loss,                    
                    price=last_price,
                    reason=decision.reason,
                )
            else:
                # fallback：原先用 strength 控制仓位
                position_mgr.open_or_add(
                    ticker=ticker,
                    strength=decision.confidence * decision.gate_mult,
                    price=last_price,
                    reason=decision.reason,
                )
            final_action = "LONG"

        elif decision.action == "REDUCE":
            reduce_size = (
                plan.size
                if (plan and plan.size > 0)
                else (decision.reduce_strength or 0.3)
            )
            position_mgr.reduce(
                ticker=ticker,
                strength=reduce_size,
                price=last_price,
                reason=decision.reason,
            )
            final_action = "REDUCE"

    # 交易完成后更新仓位信息
    pos_dict = position_mgr.pos_to_dict(ticker=ticker)
    
    return {
        "ticker": ticker,
        **pos_dict,
        **decision.__dict__,
        "atr": decision.atr,
        "action": final_action,
    }
