import numpy as np
from config.settings import MODEL_LONG_THRESHOLD, TREND_SLOPE_THRESHOLD
from strategy.calc_predicted_up import calc
from strategy.decision_context import DecisionContext
from strategy.slope import compute_slope
from strategy.strength import compute_strength
from log import signal_log
from strategy.slope import corrected_slope
from equity.regime_cooldown import regime_cooldown
from strategy.trade_intent import TradeIntent

"""
1 Gate 判断
2 模型预测
3 slope 修正
4 raw_signal
5 equity override
6 strength
7 raw_score
8 regime 合成
9 position stop/take
10 DecisionContext
"""


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

    def make_signal(self, predicted_up, slope, model_score):
        if model_score > MODEL_LONG_THRESHOLD and slope > TREND_SLOPE_THRESHOLD:
            return "LONG"



    def compute_score(
        self,
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
        eq_decision: TradeIntent,
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

        predicted_up = calc(low, median, high, latest_price)

        slope_raw = compute_slope(close_df.values)
        # 价格在涨 → 但 slope / model 仍然系统性偏空 → 这是模型结构问题,这个做短期修正
        # slope = corrected_slope(slope_raw, close_df.values[-10:])
        slope = slope_raw
        # =========================
        # 2️⃣ 原始信号
        # =========================
        if gate_result.allow:
            raw_signal = self.make_signal(predicted_up, slope, model_score)
        else:
            raw_signal = "HOLD"
        has_position = self.position_mgr.has_position(ticker)
        if has_position and eq_decision.action in ("REDUCE", "LIQUIDATE"):
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
        raw_score = self.compute_score(
            raw_signal=raw_signal,
            model_score=model_score,
            gate_mult=gate_mult,
        )

        # ========= Signal regime =========
        signal_regime = "neutral"

        if slope > 0.4 and model_score > 0.55 and gate_result.allow:
            signal_regime = "good"
        elif slope < -0.4:
            signal_regime = "bad"

        equity_regime = eq_decision.regime

        if equity_regime == "bad":
            final_regime = "bad"
        elif signal_regime == "good":
            final_regime = "good"
        else:
            final_regime = "neutral"

        pos = self.position_mgr.get(ticker)
        position_size = pos.size if pos else 0.0

        # 更新移动止损
        self.position_mgr.update_trailing_stop(ticker, latest_price, atr)

        # 止损
        liquidate_reason = None
        if pos and pos.size > 0 and latest_price <= pos.stop_loss:
            raw_signal = "LIQUIDATE"
            liquidate_reason = "STOP LOSS"

        # 止盈
        action = self.position_mgr.check_take_profit(ticker, latest_price)
        reduce_strength = eq_decision.reduce_strength
        if pos and pos.size > 0 and action == "reduce_half":
            raw_signal = "REDUCE"
            liquidate_reason = "TAKE_PROFIT"
            reduce_strength = 0.5

        dd = eq_feat["eq_drawdown"].iloc[-1] if not eq_feat.empty else 0.0
        if pos and slope > 0:
            signal_log(
                f"symbol={ticker} price={latest_price} slope={slope:.3f} model_score={model_score:.3f},atr={atr},stop_loss={pos.stop_loss} "
            )
            # signal_log(
            #     f"symbol={ticker} price={latest_price} slope={slope:.3f} model_score={model_score:.3f} raw_signal={raw_signal} final_regime={final_regime} dd={dd} reduce_strength={eq_decision.reduce_strength} gate_allow={gate_result.allow} equity_regime={eq_decision.regime} "
            # )
        ctx = DecisionContext(
            # ===== 标识 =====
            ticker=ticker,
            # ===== 市场 =====
            latest_price=latest_price,
            atr=atr,
            # ===== 模型 =====
            model_score=model_score,
            predicted_up=predicted_up,
            # ===== Gate / 动量 =====
            gate_allow=gate_result.allow,
            position_size=position_size,
            gate_mult=gate_mult,
            slope=slope,
            strength=strength,
            # ===== Regime / 确认态 =====
            regime=final_regime,
            good_count=regime_cooldown.good_count,  # ✅ 来自 regime 管理器
            good_confirm_need=regime_cooldown.good_confirm,  # ✅ 策略配置
            # ===== 冷却 =====
            regime_cooldown_left=regime_cooldown.bad_left_sec(),
            # ===== 资金 / 仓位 =====
            dd=dd,
            has_position=has_position,
            allow_add=(not has_position) and final_regime != "bad",
            # ===== 原始信号 =====
            raw_signal=raw_signal,
            raw_score=raw_score,
            reduce_strength=reduce_strength,
            liquidate_reason=liquidate_reason,
        )

        if raw_signal == "LONG":
            signal_log(
                f"ticker={ticker} med={median[-1]}, price={latest_price:.2f}, pre_up={predicted_up:.3f}, slope={slope:.3f} model_score={model_score:.3f} LONG med:date={close_df.index[-1].strftime('%Y-%m-%d %H:%M')}, "
            )
            # signal_log(ctx)
        return ctx
