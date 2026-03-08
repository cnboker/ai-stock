import numpy as np
from strategy.decision_context import DecisionContext
from strategy.slope import compute_slope
from strategy.strength import compute_strength
from log import signal_log
from strategy.slope import corrected_slope


def make_signal(predicted_up, slope, model_score):

    if model_score > 0.56 and predicted_up > 0.004 and slope > 0.1:
        return "LONG"

    if model_score < 0.44 and predicted_up < -0.004 and slope < -0.1:
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
    model_score = abs(model_score)
    if raw_signal == "LONG":
        direction = 1.0

    elif raw_signal == "SHORT":
        direction = -1.0

    elif raw_signal == "HOLD":
        direction = 0.0
        model_score *= hold_penalty

    else:
        return 0.0

    score = direction * model_score * gate_mult
    return float(np.clip(score, -1.0, 1.0))


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
        close_df,
        eq_decision,
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
        
        predicted_up = self.equity_engine.calc_predicted_up_risk_adjusted(
            low, median, high, latest_price
        )
       
        slope_raw = compute_slope(
            predicted_up=predicted_up,
            horizon=1,
        )
        # 价格在涨 → 但 slope / model 仍然系统性偏空 → 这是模型结构问题,这个做短期修正
        slope = corrected_slope(slope_raw, close_df.values[-10:])
        if median[-1]> latest_price:
            print(f"med:{median[-1]}, price={latest_price}, pre_up={predicted_up}, slope={slope}")
        # =========================
        # 2️⃣ 原始信号
        # =========================
        if gate_result.allow:
            raw_signal = make_signal(predicted_up, slope, model_score)
        else:
            raw_signal = "HOLD"
        has_position = self.position_mgr.has_position(ticker)
        if eq_decision.action and has_position:
            raw_signal = eq_decision.action

        # =========================
        # 4️⃣ Gate / slope / strength
        # =========================
        gate_mult = eq_decision.gate_mult

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

        allow_add = not has_position
        dd = 0
        if not eq_feat.empty:
            row = eq_feat.iloc[-1]
            dd = float(row["eq_drawdown"])
        cool_mgr = self.equity_engine.cooldown_mgr
        if strength > 0.7 and gate_result.allow:
            new_regime = "good"
        elif strength < 0.2:
            new_regime = "bad"
        else:
            new_regime = "neutral"
        regime = cool_mgr.update(new_regime)
        # signal_log(
        #     f"regime_update new={new_regime} "
        #     f"regime={cool_mgr.regime} "
        #     f"good={cool_mgr.good_count}/{cool_mgr.good_confirm}"
        # )

        pos = self.position_mgr.get(ticker)
        position_size = pos.size if pos else 0.0

        action_signal = self.position_mgr.check_stop_take(ticker, latest_price)
        if action_signal:
            raw_signal = "LIQUIDATE"

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
            position_size=position_size,
            gate_mult=gate_mult,
            slope=slope,
            strength=strength,
            # ===== Regime / 确认态 =====
            regime=regime,
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
