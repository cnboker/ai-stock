from typing import Optional
import numpy as np
from global_state import equity_engine
from strategy.equity_policy import TradeIntent

def make_signal(
    low: np.ndarray,
    median: np.ndarray,
    high: np.ndarray,
    latest_price: float,
    up_thresh: float = 0.006,  # +0.6%
    down_thresh: float = -0.006,  # -0.6%
):
    if len(median) == 0:
        return "HOLD"

    pred_mid = median[-1]
    pred_low = low[-1]
    pred_high = high[-1]

    up_ratio = (pred_high - latest_price) / latest_price
    down_ratio = (pred_low - latest_price) / latest_price
    mid_ratio = (pred_mid - latest_price) / latest_price

    if mid_ratio > up_thresh and down_ratio > -0.003:
        return "LONG"

    if mid_ratio < down_thresh and up_ratio < 0.003:
        return "SHORT"

    return "HOLD"


def compute_score(raw_signal, model_score, eq_decision: TradeIntent, gate_mult=1.0):
    if raw_signal == "LONG":
        return +model_score * gate_mult
    elif raw_signal == "SHORT":
        return -model_score * gate_mult
    elif raw_signal in ("REDUCE", "LIQUIDATE"):
        return -eq_decision.reduce_strength * gate_mult
    else:
        return 0.0

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
   
    def __init__(self, gater, debouncer_manager, min_score: float = 0.05):
        self.gater = gater
        self.debouncer = debouncer_manager        
        self.min_score = min_score

    def evaluate(
        self,
        *,
        ticker: str,
        low: np.ndarray,
        median: np.ndarray,
        high: np.ndarray,
        latest_price: float,
        close_df,
        model_score: float,
        eq_decision: TradeIntent,
        has_position:bool,
        atr: float = 1.0
    ) -> TradeIntent:

        # =========================
        # 1️⃣ Gate 逻辑
        # =========================
        gate_result = self.gater.evaluate(
            lower=low,
            mid=median,
            upper=high,
            close_df=close_df.values
        )

        if not gate_result.allow:
            raw_signal = "HOLD"
        else:
            raw_signal = make_signal(low, median, high, latest_price)
        print(
            ticker,
            "gate_allow=", gate_result.allow,
            "raw_from_model=", raw_signal
        )
        # =========================
        # 2️⃣ 当前股票有仓位执行减仓决策
        # =========================        
        if eq_decision.action and has_position:
            raw_signal = eq_decision.action

        gate_mult = eq_decision.gate_mult

        # =========================
        # 3️⃣ Score 计算
        # =========================
        final_score = compute_score(raw_signal, model_score, eq_decision, gate_mult)

        # =========================================================
        # 4️⃣ 弱信号过滤（升级版）
        # =========================================================
        # 规则：
        # 1. LONG 趋势允许，即便 score < min_score
        # 2. SHORT / REDUCE 按原逻辑
        # 3. 保留 min_score 对 HOLD 的判断
        if raw_signal not in ("REDUCE", "LIQUIDATE"):
            if raw_signal == "LONG" and (model_score > 0.01):
                pass
            elif abs(final_score) < self.min_score:
                final_score = 0.0

        # =========================
        # 4️⃣ Debouncer
        # =========================
        final_action, confidence = self.debouncer.update(ticker, final_score, atr=atr)
        confirmed = final_action != "HOLD" and confidence > 0


        return TradeIntent(
            action=final_action,
            raw_action = eq_decision.raw_action, #未执行减仓策略的时候的action
            confidence=confidence,
            regime=eq_decision.regime,
            gate_mult=gate_mult,
            force_reduce=eq_decision.force_reduce,
            reduce_strength=eq_decision.reduce_strength,
            raw_score=final_score,
            confirmed=confirmed,
            reason=f"raw={raw_signal}, score={final_score:.3f}"
        )