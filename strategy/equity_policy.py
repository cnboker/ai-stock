from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime
from strategy.signal_debouncer import debouncer_manager
from strategy.trade_intent import TradeIntent

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

def drawdown_level(dd: float) -> int:
    if dd >= 0.08:
        return 4
    if dd >= 0.06:
        return 3
    if dd >= 0.03:
        return 2
    if dd >= 0.01:
        return 1
    return 0



'''
系统行为
回撤	level	动作
-1.5%	0	无
-2.3%	1	REDUCE 0.3
-2.1%	1	❌ 不再触发
-3.0%	1	❌ 不再触发
-5.4%	2	REDUCE 0.6
-5.1%	2	❌ 不再触发
'''
#同一等级，只触发一次
def reduce_policy_with_guard(drawdown, last_level):
    dd = abs(drawdown)
    level = drawdown_level(dd)

    # 没升级 → 什么都不做
    if level <= last_level:
        return last_level, False, 0.0

    # 升级
    strength_map = {
        1: 0.3,
        2: 0.6,
        3: 1.0,
        4: 1.0,
    }

    return level, True, strength_map.get(level, 0.0)


# Equity 风控 + 状态裁决器
def decide_equity_policy(eq_feat, has_position: bool, equity_state) -> TradeIntent:
    
    if eq_feat is None or eq_feat.empty:
        return TradeIntent(
            action="HOLD",
            regime="neutral",
            gate_mult=1.0,
        )

    regime = equity_regime(eq_feat)
    gate_mult = equity_gate(eq_feat)

    action = "HOLD"
    reduce_strength = 0.0
    reason = ""

    if has_position:
        dd = eq_feat["eq_drawdown"].iloc[-1]

        level, should_reduce, reduce_strength = reduce_policy_with_guard(
            drawdown=dd,
            last_level=equity_state.dd_level,
        )
       
        if should_reduce:
            equity_state.dd_level = level
            action = "REDUCE"   # ✅ 这里必须是字符串
            reason = f"eq_drawdown_level_{level}"

    return TradeIntent(
        action=action,
        regime=regime,
        gate_mult=gate_mult,
        reduce_strength=reduce_strength,
        reason=reason,
        has_position=has_position,
    )


def decision_from_score(
    *,
    ticker: str,
    score: float,
    atr: float,
    regime: str,
) -> TradeIntent:
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

    return TradeIntent(
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
