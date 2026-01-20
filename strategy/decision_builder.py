import numpy as np
from strategy.decision_context import DecisionContext
from strategy.slope import compute_slope
from strategy.strength import compute_strength
from log import signal_log
from strategy.slope import corrected_slope
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


"""
HOLD 不再是 0

model_score 永远生效

gate_mult 自然降权

raw_score 不会全灭
"""


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
        direction = +1.0  # 有预测方向，但弱
        model_score *= hold_penalty
    else:
        return 0.0

    # ---------- 强度 ----------
    score = direction * model_score * gate_mult
    score = float(np.clip(score, -1.0, 1.0))

    return score


class DecisionContextBuilder:
    def __init__(
        self,
        *,
        equity_engine,
        gater,
        position_mgr,
    ):
        self.equity_engine = equity_engine
        self.gater = gater
        self.position_mgr = position_mgr


    def build(
        self,
        *,
        ticker: str,
        low,
        median,
        high,
        latest_price: float,
        atr: float,
        model_score: float,
        eq_feat,
        close_df        
    ) -> DecisionContext:

        # =========================
        # 1️⃣ Gate 判断（结构）
        # =========================
        gate_result = self.gater.evaluate(
            lower=low,
            mid=median,
            upper=high,
            close_df=close_df.values,
        )

        # =========================
        # 2️⃣ 原始信号
        # =========================
        if gate_result.allow:
            raw_signal = make_signal(low, median, high, latest_price)
        else:
            raw_signal = "HOLD"

        # =========================
        # 3️⃣ Equity 决策
        # =========================
        eq_decision = self.equity_engine.decide(
            eq_feat=eq_feat,
            has_position=self.position_mgr.has_position(ticker),
        )
        
        if eq_decision.action and self.position_mgr.has_position(ticker):
            raw_signal = eq_decision.action
        #signal_log(eq_decision)
        # =========================
        # 4️⃣ Gate / slope / strength
        # =========================
        gate_mult = eq_decision.gate_mult

        slope_raw = compute_slope(
            predicted_up=self.equity_engine.calc_predicted_up_risk_adjusted(
                low, median, high, latest_price
            ),
            horizon=1,
        )
        #价格在涨 → 但 slope / model 仍然系统性偏空 → 这是模型结构问题,这个做短期修正
        slope = corrected_slope(slope_raw, close_df.values[-10:]) 
        strength = compute_strength(
            slope=slope,
            gate=gate_mult,
        )

        # =========================
        # 5️⃣ raw_score
        # =========================
        raw_score = compute_score(
            raw_signal=raw_signal,
            model_score=model_score,
            gate_mult=gate_mult,
        )

        # =========================
        # 6️⃣ 仓位 / 回撤
        # =========================
        has_position = self.position_mgr.has_position(ticker)
        allow_add = not has_position

        row = eq_feat.iloc[-1]
        dd = float(row["eq_drawdown"])
        cool_mgr = self.equity_engine.cooldown_mgr
        pos = self.position_mgr.get(ticker)
        position_size = pos.size if pos else 0.0
        return DecisionContext(
            # ===== 标识 =====
            ticker=ticker,
            # ===== 市场 =====
            latest_price=latest_price,
            atr=atr,
            # ===== 模型 =====
            model_score=model_score,
            predicted_up=self.equity_engine.calc_predicted_up_risk_adjusted(
                low, median, high, latest_price
            ),
            # ===== Gate / 动量 =====
            gate_allow=gate_result.allow,
            position_size= position_size,
            gate_mult=gate_mult,
            slope=slope,
            strength=strength,
            # ===== Regime / 确认态 =====
            regime=eq_decision.regime,
            good_count=cool_mgr.good_count,  # ✅ 来自 regime 管理器
            good_confirm_need=cool_mgr.good_confirm,  # ✅ 策略配置
            # ===== 冷却 =====
            regime_cooldown_left=cool_mgr.bad_left_sec(),
            # ===== 资金 / 仓位 =====
            dd=dd,
            has_position=has_position,
            allow_add=allow_add,
            # ===== 原始信号 =====
            raw_signal=raw_signal,
            raw_score=raw_score,
        )
