from datetime import datetime
from typing import Optional
import numpy as np
from pandas import DataFrame
from equity.equity_gate import compute_gate
from log import get_logger
from strategy.decision_context import DecisionContext
from strategy.hold_reason import decide_good_hold_reason
from strategy.trade_intent import TradeIntent

"""
价格预测（Chronos）
        ↓
结构判断（gater）
        ↓
资金状态审查（equity_regime）
        ↓
资金状态放大/压制（equity_gate）
        ↓
信号稳定器（debouncer）


    你现在已经具备的能力

    ✔ bad 状态自动冷却

    ✔ 连续 good 才放行

    ✔ 分级减仓 / 强平

    ✔ Equity 决策集中管理

    ✔ SignalManager 干净可维护

    raw_signal	含义
    LONG	尝试开/加多
    SHORT	尝试开/加空
    HOLD	不新增风险
    REDUCE	主动减仓
    LIQUIDATE	强平

 
    信号管理器，处理 Gate / Debouncer / 最终决策
    """


class SignalManager:
    def __init__(self, debouncer, min_score: float = 0.01):
        self.debouncer = debouncer
        self.min_score = min_score

    def evaluate(self, ctx: DecisionContext) -> TradeIntent:
        # =========================
        # 1️⃣ 弱信号过滤（不碰 REDUCE / LIQUIDATE）
        # =========================
        score = ctx.raw_score

        if ctx.raw_signal not in ("REDUCE", "LIQUIDATE"):
            if abs(score) < self.min_score:
                score = 0.0

            if ctx.raw_signal == "HOLD":
                score = 0.0

        # =========================
        # 2️⃣ 去抖动确认
        # =========================
        action, confidence, state = self.debouncer.update(
            ctx.ticker, score, atr=ctx.atr
        )

        confirmed = action != "HOLD" and confidence > 0
        good_hold_reason = None
        good_hold_detail = {}

        if (
            ctx.regime == "good"
            and action == "HOLD"
            and ctx.raw_signal != "HOLD"
        ):
            good_hold_reason, good_hold_detail = decide_good_hold_reason(ctx)

        force_reduce = ctx.raw_signal in ("REDUCE", "LIQUIDATE")

        # =========================
        # 3️⃣ 构造 TradeIntent
        # =========================
        intent = TradeIntent(            
            action=action,
            confidence=confidence,
            confirmed=confirmed,
            raw_action=ctx.raw_signal,
            raw_score=ctx.raw_score,
            model_score=ctx.model_score,
            predicted_up=ctx.predicted_up,
            strength=ctx.strength,
            has_position=ctx.has_position,
            force_reduce=force_reduce,
            regime=ctx.regime,
            gate_allow=ctx.gate_allow,
            gate_mult=ctx.gate_mult,
            reason=(
                f"raw={ctx.raw_signal}, "
                f"raw_score={ctx.raw_score:.3f}, "
                f"final_score={score:.3f}"
            ),
            good_hold_reason=good_hold_reason,
            good_hold_detail=good_hold_detail
        )

        # =========================
        # 4️⃣ 冷却状态标记（只读）
        # =========================
        intent.cooldown_active = action == "HOLD" and ctx.raw_signal != "HOLD"
        intent.cooldown_left = max(0, state.confirm_n - state.count)

        return intent
