import pandas as pd
from strategy.equity_policy import TradeIntent

'''
典型触发场景:decision.force_reduce
场景	触发条件	解释
止损	股票价格跌破某个止损价	系统发现当前持仓亏损过大，需要强制减仓，忽略模型动作
风险控制	当日总亏损/浮动亏损超过阈值	风控策略限制账户整体风险，所有触发仓位可能被强制减仓
仓位上限	当前持仓超过策略允许最大仓位	强制减仓以回到安全范围
异常事件	股票停牌、数据异常、行情异常	系统检测到异常，立刻减仓或清仓
策略切换	新的市场 regime 切换，当前策略不适用	强制降低仓位或平掉不安全的仓位
'''
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
    print('final_action',final_action, decision.action,decision.force_reduce)
    # 统一返回字典
    return {
        "ticker": ticker,
        "position": position_mgr.get(ticker),
        "action": final_action,
        "confidence": getattr(decision, "confidence", 0),
        "model_score": getattr(decision, "model_score", 0),
        "atr": getattr(decision, "atr", 0),
        "predicted_up": getattr(decision, "predicted_up", False),
        "raw_score": getattr(decision, "raw_score", 0),
        "regime": decision.regime
    }

def decision_to_dict(decision):
    pos = decision["position"]
    return {
        "ticker": decision["ticker"],
        "position_size": getattr(pos, "size", 0),
        "direction": getattr(pos, "direction", ""),
        "entry_price": getattr(pos, "entry_price", 0),
        "stop_loss": getattr(pos, "stop_loss", 0),
        "take_profit": getattr(pos, "take_profit", 0),
        "action": decision.get("action"),
        "force_reduce": decision.get("force_reduce", False),
        "confidence": decision.get("confidence", 0),
        "model_score": decision.get("model_score", 0),
        "atr": decision.get("atr", 0),
        "predicted_up": decision.get("predicted_up", False),
        "raw_score": decision.get("raw_score", 0),
        "regime":decision.get("regime", 0),
        "gate_mult":decision.get("gate_mult", 0.0),
        "confirmed": decision.get("confirmed",False)
    }