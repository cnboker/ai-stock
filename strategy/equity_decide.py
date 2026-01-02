from dataclasses import dataclass

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime
'''
action: str

表示具体交易动作。通常是：

"BUY" / "LONG" → 建仓或加仓

"SELL" / "SHORT" → 减仓或做空

"HOLD" → 不操作

这个是策略直接输出的信号核心。

regime: str

表示当前市场或策略的宏观判断，影响策略的激进/保守程度。

常用值：

"good" → 市场/信号良好，可以积极建仓

"neutral" → 中性，按常规仓位操作

"bad" → 市场不好，降低仓位或者延迟开仓

gate_mult: float

仓位放大/压制系数。

用于调整实际建仓量：

target_position = base_position * gate_mult


例子：

gate_mult=1.0 → 仓位不变

gate_mult=0.5 → 只建半仓

gate_mult=2.0 → 加倍仓位

force_reduce: bool

是否强制减仓，即不管策略原信号如何，都强制减仓。

常用在：

系统风控触发

风险过高时

reduce_strength: float

减仓力度，取值 0~1

0 → 不减仓

0.5 → 减半仓

1.0 → 全部平仓

当 force_reduce=True 时，这个值通常会被策略读取来执行减仓。
'''
@dataclass
class EquityDecision:
    action:str
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
            action="HOLD",
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
