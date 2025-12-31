from dataclasses import dataclass

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime

@dataclass
class EquityDecision:
    regime: str              # good / neutral / bad
    gate_mult: float         # 仓位放大/压制
    force_reduce: bool       # 是否强制减仓
    reduce_strength: float   # 0~1

#分级减仓 / 强平策略
def reduce_policy(drawdown: float):
    """
    drawdown: 负数
    """
    dd = abs(drawdown)

    if dd >= 0.10:
        return "LIQUIDATE", 1.0 #强平
    if dd >= 0.08:
        return "REDUCE", 1.0
    if dd >= 0.05:
        return "REDUCE", 0.6
    if dd >= 0.02:
        return "REDUCE", 0.3

    return None, 0.0

#统一 Equity 决策入口
def equity_decide(eq_feat, has_position: bool) -> EquityDecision:
    # ========= 默认值 =========
    if eq_feat is None or eq_feat.empty:
        return EquityDecision(
            regime="neutral",
            gate_mult=1.0,
            force_reduce=False,
            reduce_strength=0.0,
        )

    # ========= regime =========
    regime = equity_regime(eq_feat)

    # ========= gate =========
    gate_mult = equity_gate(eq_feat)

    # ========= reduce =========
    # ========= 风控 =========
    action = None
    reduce_strength = 0.0

    if has_position:
        dd = eq_feat["eq_drawdown"].iloc[-1]
        action, reduce_strength = reduce_policy(dd)


    return EquityDecision(
        regime=regime,
        gate_mult=gate_mult,
        action=action,
        reduce_strength=reduce_strength,
    )
