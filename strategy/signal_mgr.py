from strategy.decision_context import DecisionContext
from strategy.trade_intent import TradeIntent
from log import signal_log

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
        score = ctx.raw_score
        if ctx.regime == "bad":
            return TradeIntent(confirmed=False, reason="bad regime")
        # =========================
        # 0️⃣ 风控优先级最高（直接 bypass debounce）
        # =========================
        if ctx.raw_signal in ("REDUCE", "LIQUIDATE"):
            return TradeIntent(
                action=ctx.raw_signal,
                confidence=1.0,
                confirmed=True,
                atr=ctx.atr,
                raw_action=ctx.raw_signal,
                raw_score=ctx.raw_score,
                model_score=ctx.model_score,
                predicted_up=ctx.predicted_up,
                strength=ctx.strength,
                force_reduce=True,
                reduce_strength=ctx.reduce_strength or 1.0,
                regime=ctx.regime,
                gate_allow=ctx.gate_allow,
                gate_mult=ctx.gate_mult,
                reason= ctx.liquidate_reason
            )

        # =========================
        # 1️⃣ 弱信号过滤（只处理非强制信号）
        # =========================
        if abs(score) < self.min_score:
            score = 0.0

        if ctx.raw_signal == "HOLD":
            score = 0.0

        #signal_log(f"ctx.raw_signal={ctx.raw_signal},raw_score={ctx.raw_score},")

        # =========================
        # 2️⃣ Debounce
        # =========================
        action, confidence, state = self.debouncer.update(
            ctx.ticker, score, atr=ctx.atr
        )

       # 【核心修改点】：如果在 good regime 下，raw_signal 是 LONG，且已经有一定强度
        # 我们不等 debouncer 的计数器跑完，直接“强制确认”
        is_trend_confirmed = False
        if ctx.regime == "good" and ctx.raw_signal == "LONG":
            if ctx.model_score > self.min_score * 5: # 设定一个合理的起步分
                is_trend_confirmed = True

        confirmed = (action != "HOLD" and confidence > 0) or is_trend_confirmed

        intent = TradeIntent(
            action="LONG" if is_trend_confirmed else action, # 强制修正 action
            confidence=max(confidence, ctx.model_score) if is_trend_confirmed else confidence,
            confirmed=confirmed,
            raw_action=ctx.raw_signal,
            raw_score=ctx.raw_score,
            model_score=ctx.model_score,
            predicted_up=ctx.predicted_up,
            strength=ctx.strength,
            atr=ctx.atr,
         
            force_reduce=False,
            reduce_strength=confidence,
            regime=ctx.regime,
            gate_allow=ctx.gate_allow,
            gate_mult=ctx.gate_mult,
            reason=(
                f"raw={ctx.raw_signal}, "
                f"raw_score={ctx.raw_score:.3f}, "
                f"final_score={score:.3f}"
            ),
        )


        return intent