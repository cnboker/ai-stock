import numpy as np
from strategy.calc_predicted_up import calc
from strategy.decision_context import DecisionContext
from strategy.slope import compute_slope
from log import signal_log
from strategy.strength import compute_strength
from strategy.trade_intent import TradeIntent
from infra.core.dynamic_settings import settings


class DecisionContextBuilder:
    def __init__(self, *, gater, position_mgr):
        self.gater = gater
        self.position_mgr = position_mgr

    def _make_raw_signal(self, predicted_up, slope, model_score):
        """基于模型输出生成初步信号"""
        if (
            model_score > settings.MODEL_TH
            and slope > settings.SLOPE
            and predicted_up > settings.PREDICT_UP
        ):
            return "LONG"
        return "HOLD"

    def _compute_debounce_score(self, raw_signal, model_score, gate_mult):
        """计算给 SignalManager (Debouncer) 使用的原始分"""
        model_score = abs(model_score)
        direction = 0.0
        if raw_signal == "LONG":
            direction = 1.0
        elif raw_signal == "SHORT":
            direction = -1.0
        elif raw_signal == "HOLD":
            # HOLD 状态给予一定惩罚，防止反复震荡
            direction = 0.0
            model_score *= 0.3

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
        close_df,
        eq_decision: TradeIntent,  # 外部 EquityRiskEngine 传入
    ) -> DecisionContext:

        # 1. 结构筛选 (Gater)
        gate_result = self.gater.evaluate(
            lower=low, mid=median, upper=high, close_df=close_df.values
        )

        # 2. 基础计算
        predicted_up = calc(low, median, high, latest_price)
        slope = compute_slope(close_df.values)

        # 3. 初始信号判定
        raw_signal = (
            self._make_raw_signal(predicted_up, slope, model_score)
            if gate_result.allow
            else "HOLD"
        )

        # 4. 账户状态同步 (Regime 合成)
        # 策略：只要账户风险层说是 bad，整体就是 bad；否则看信号动量

        is_trend_strong = slope > settings.SLOPE
        is_model_confident = model_score > settings.MODEL_TH
        is_gate_open = gate_result.allow

        # 优化后的判定逻辑
        signal_regime = "neutral"
        
        # 1. 完美状态：全部达标
        if is_trend_strong and is_model_confident and is_gate_open:
            signal_regime = "good"
        
        # 2. 趋势初期：模型极度自信 + 门槛通过 (即便斜率还没完全拉起)
        elif model_score > (settings.MODEL_TH + 0.1) and is_gate_open:
            signal_regime = "good"
            
        # 3. 强势修正：斜率极好 + 门槛通过 (即便模型分刚过线)
        elif slope > (settings.SLOPE * 1.5) and is_gate_open:
            signal_regime = "good"

        elif slope < -0.4:
            signal_regime = "bad"

        final_regime = "bad" if eq_decision.regime == "bad" else signal_regime

        # 5. 持仓状态处理 (止损/止盈)
        pos = self.position_mgr.get(ticker)
        has_position = pos is not None
        position_size = pos.size if has_position else 0.0

        liquidate_reason = None
        reduce_strength = eq_decision.reduce_strength  # 初始强度来自账户风控

        # 更新移动止损并检查
        self.position_mgr.update_trailing_stop(ticker, latest_price, atr)

        if has_position:
            # A. 检查止损
            if latest_price <= pos.stop_loss:
                latest_price = pos.stop_loss * 0.998  # 模拟滑点成交
                raw_signal = "LIQUIDATE"
                liquidate_reason = "STOP LOSS"
                self.position_mgr.cooldown[ticker] = 3

            # B. 检查账户风险指令 (REDUCE/LIQUIDATE)
            elif eq_decision.action in ("REDUCE", "LIQUIDATE"):
                raw_signal = eq_decision.action
                liquidate_reason = eq_decision.reason

            # C. 检查个股止盈
            tp_action = self.position_mgr.check_take_profit(ticker, latest_price)
            if isinstance(tp_action, float):
                raw_signal = "REDUCE"
                # 取账户减仓要求和个股止盈强度的最大值 (取严原则)
                reduce_strength = max(float(reduce_strength or 0.0), float(tp_action or 0.0))
                liquidate_reason = "TAKE_PROFIT"
                self.position_mgr.cooldown[ticker] = 3

        # 6. 计算最终缩放系数与分数
        final_gate_mult = gate_result.score * eq_decision.gate_mult
        raw_score = self._compute_debounce_score(
            raw_signal, model_score, final_gate_mult
        )

        # 7. 善后处理
        self.position_mgr.update_cooldown()
        #strength = compute_strength(slope=slope, gate=final_gate_mult)
        strength = compute_strength(
            slope=slope,
            gate=final_gate_mult,
            alpha=settings.STRENGTH_ALPHA,  # 确保是从动态配置对象里取的
            slope_min=settings.SLOPE_MIN    # 同上
        )
        ctx = DecisionContext(
            ticker=ticker,
            latest_price=latest_price,
            atr=atr,
            model_score=model_score,
            predicted_up=predicted_up,
            gate_allow=bool(gate_result.allow),
            gate_mult=final_gate_mult,
            regime=final_regime,
            has_position=has_position,
            position_size=position_size,
            raw_signal=raw_signal,
            raw_score=raw_score,
            reduce_strength=reduce_strength,
            liquidate_reason=liquidate_reason,
            strength=strength, #大于0才有开仓意图
            slope=slope,
        )

        # if raw_signal == "LONG":
        #     print(f"settings.STRENGTH_ALPHA: {settings.STRENGTH_ALPHA}")
        #     signal_log(ctx)
        #if raw_signal == "LONG":
        # signal_log(
        #     f"🔥 {ticker} |eq_decision.action={eq_decision.action} Lost_price={pos.stop_loss if pos else None} raw_signal={raw_signal} | final_regime:{final_regime} | Price: {latest_price:.2f} | "
        #     f"Pre_Up: {predicted_up:.3f} | Score: {model_score:.3f} | Gate_Mult: {final_gate_mult:.2f}"
        # )
        #signal_log(ctx)
        return ctx
