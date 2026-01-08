from typing import Optional
import numpy as np
from log import get_logger
from strategy.equity_policy import TradeIntent
from dataclasses import replace

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

'''
HOLD 不再是 0

model_score 永远生效

gate_mult 自然降权

raw_score 不会全灭
'''
def compute_score(
    *,
    raw_signal: str,
    model_score: float,
    gate_mult: float = 1.0,
    hold_penalty: float = 0.3,
) -> float:
    """
    返回 [-1, 1] 的连续 score
    """

    # ---------- 方向 ----------
    if raw_signal == "LONG":
        direction = +1.0
    elif raw_signal == "SHORT":
        direction = -1.0
    elif raw_signal == "HOLD":
        direction = +1.0   # 有预测方向，但弱
        model_score *= hold_penalty
    else:
        return 0.0

    # ---------- 强度 ----------
    score = direction * model_score * gate_mult
    score =  float(np.clip(score, -1.0, 1.0))

    return score


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
   
    def __init__(self, gater, debouncer_manager, min_score: float = 0.01):
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
       
        # =========================
        # 2️⃣ 当前股票有仓位执行减仓决策
        # =========================        
        if eq_decision.action and has_position:
            raw_signal = eq_decision.action

        gate_mult = eq_decision.gate_mult

        # =========================
        # 3️⃣ Score 计算
        # =========================
        # 1️⃣ 计算原始分数
        raw_score_before_filter = compute_score(
            raw_signal=raw_signal,
            model_score=model_score,
            gate_mult=gate_mult
        )


        # 弱信号过滤（升级版）
        final_score = raw_score_before_filter
        if raw_signal not in ("REDUCE", "LIQUIDATE"):
            if raw_signal == "LONG":
                if model_score > 0.01:
                    # 微弱 LONG 信号保留
                    pass
                else:
                    final_score = 0.0
            elif raw_signal == "SHORT":
                # 可根据需要保留微弱 SHORT，这里按 min_score 过滤
                if abs(final_score) < self.min_score:
                    final_score = 0.0
            else:
                # HOLD 一律置零
                final_score = 0.0

        # 3️⃣ 去抖动 Debouncer
        final_action, confidence = self.debouncer.update(ticker, final_score, atr=atr)
        confirmed = final_action != "HOLD" and confidence > 0

        # 4️⃣ 打印调试日志
        get_logger("signal").info(
            f"[SIGNAL DEBUG] {ticker}: gate_allow= {gate_result.allow}, "
            f"raw_signal={raw_signal}, "
            f"compute_score={raw_score_before_filter:.6f}, "
            f"after_weak_filter={final_score:.6f}, "
            f"final_action={final_action}, confidence={confidence:.3f}"
        )

        # 5️⃣ 返回更新后的决策
        return replace(
            eq_decision,
            action=final_action,
            confidence=confidence,
            gate_mult=gate_mult,
            raw_score=final_score,
            confirmed=confirmed,
            reason=f"raw={raw_signal}, compute_score={raw_score_before_filter:.3f}, final_score={final_score:.3f}"
        )
