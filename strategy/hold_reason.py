from strategy.decision_context import DecisionContext, GoodHoldReason


def decide_good_hold_reason(
    ctx: DecisionContext,
) -> tuple[GoodHoldReason, dict]:

    # 1️⃣ 冷却（硬约束）
    if ctx.regime == "bad" and ctx.regime_cooldown_left > 0:
        return (
            GoodHoldReason.COOLDOWN_ACTIVE,
            {"left_sec": ctx.regime_cooldown_left}
        )

    # 2️⃣ regime 未确认
    if ctx.regime != "good":
        return (
            GoodHoldReason.CONFIRMATION_PENDING,
            {
                "regime": ctx.regime,
                "good_count": ctx.good_count,
                "good_need": ctx.good_confirm_need,
            }
        )

    # 3️⃣ 仓位限制
    if ctx.has_position and not ctx.allow_add:
        return (
            GoodHoldReason.POSITION_LIMIT,
            {"position_size": ctx.position_size}
        )

    # 4️⃣ gate 抑制
    if not ctx.gate_allow or ctx.raw_score < ctx.gate:
        return (
            GoodHoldReason.GATE_NOT_PASSED,
            {
                "raw_score": ctx.raw_score,
                "gate": ctx.gate,
            }
        )

    # 5️⃣ strength 不够
    if ctx.strength <= 0.01:
        return (
            GoodHoldReason.STRENGTH_TOO_WEAK,
            {"strength": ctx.strength}
        )

    # 6️⃣ 趋势衰减
    if ctx.slope < ctx.slope_decay_thresh:
        return (
            GoodHoldReason.TREND_DECAY,
            {"slope": ctx.slope}
        )

    return GoodHoldReason.NONE, {}
