from dataclasses import dataclass

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime
from strategy.signal_debouncer import debouncer_manager

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

    🔴 force_reduce 应该 只在这 3 种情况下为 True

        1️⃣ 系统级风控

        回撤超过阈值

        连续亏损

        Equity slope 崩坏

        2️⃣ 异常状态

        数据缺失

        模型崩溃

        时间轴断裂

        3️⃣ 人工干预

        手动触发降仓

        临时避险

'''
@dataclass
class TradeContext:
    action:str                # ->执行意图  
    regime: str              # good / neutral / bad 资金/->风险状态
    gate_mult: float         # 仓位放大/压制 ->风控调制
    reduce_strength: float   # 0~1 执行参数
    force_reduce: bool = False      # 是否强制减仓 ->强制约束
   
    # ===== 信号语义层（新增）=====
    confidence: float = 0.0     # 事件强度（来自 debouncer）->信号质量
    raw_score: float = 0.0      # 连续 score（模型 × gate × equity）->模型派生
    model_score: float = 0.0
    confirmed: bool = False    # 是否通过 debouncer ->稳定器结果
    reason: str = ""            # 触发原因（日志 / 回测用）
    atr: float = 0.0

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
def decide_equity_policy(eq_feat, has_position: bool) -> TradeContext:
    # ========= 默认值 =========
    if eq_feat is None or eq_feat.empty:
        return TradeContext(
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
    action = None
    reduce_strength = 0.0

    if has_position:
        dd = eq_feat["eq_drawdown"].iloc[-1]
        action, reduce_strength = reduce_policy(dd)


    return TradeContext(
        regime=regime,
        gate_mult=gate_mult,
        action=action,
        reduce_strength=reduce_strength,
    )


def decision_from_score(
    *,
    ticker: str,
    score: float,
    atr: float,
    regime: str,
) -> TradeContext:
    """
    把模型 score + debouncer 输出，转成唯一交易决策对象
    """

    # ===== 1. Debounce =====
    action, confidence = debouncer_manager.update(
        ticker=ticker,
        final_score=score,
        atr=atr,
    )

    confirmed = confidence > 0

    # ===== 2. gate_mult：是否放大仓位 =====
    if regime == "good":
        gate_mult = 1.0
    elif regime == "neutral":
        gate_mult = 0.5
    else:
        gate_mult = 0.0

    # ===== 3. 强制风控（bad regime）=====
    force_reduce = regime == "bad"
    reduce_strength = 1.0 if force_reduce else confidence

    # ===== 4. reason 解释 =====
    if force_reduce:
        reason = "regime_bad_force_reduce"
    elif not confirmed:
        reason = "debounce_not_confirmed"
    else:
        reason = f"signal_confirmed_{action.lower()}"

    return TradeContext(
        action=action,
        regime=regime,
        gate_mult=gate_mult,
        force_reduce=force_reduce,
        reduce_strength=reduce_strength,
        confidence=confidence,
        raw_score=score,
        confirmed=confirmed,
        reason=reason,
    )
