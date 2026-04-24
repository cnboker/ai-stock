import numpy as np
from infra.core.runtime import GlobalState
from strategy.calc_predicted_up import calc
from strategy.decision_context import DecisionContext
from strategy.slope import compute_hybrid_slope
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
        print(
            f"settings.MODEL_TH: {settings.MODEL_TH}, settings.SLOPE: {settings.SLOPE}, settings.PREDICT_UP: {settings.PREDICT_UP}"
        )
        is_momentum_long = (
            model_score > settings.MODEL_TH
            and slope > settings.SLOPE
            and predicted_up > settings.PREDICT_UP
        )

        # 逻辑 2：超跌反弹捕获 (新增)
        # 当分数极高且预期涨幅很大时，即便斜率刚抬头(>0)也可以考虑
        is_rebound_long = model_score > 0.9 and predicted_up > 0.02 and slope > 0

        if is_momentum_long or is_rebound_long:
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
        std: np.ndarray = None,  # 新增 std 接口
        atr: float,
        model_score: float,
        close_df,
        eq_decision: TradeIntent,  # 外部 EquityRiskEngine 传入
    ) -> DecisionContext:
        latest_price = GlobalState.tickers_price[ticker]
        # 1. 结构筛选 (Gater)
        gate_result = self.gater.evaluate(
            lower=low, mid=median, upper=high, close_df=close_df.values
        )
        # 【进化点】风险过滤：如果预测的波动率异常放大，说明模型心里也没底
        if std is not None:
            # 计算预测的相对波动率 (Coefficient of Variation)
            # 这里取未来几个预测点的平均 std / median
            relative_vol = (std / median).mean()

            # 如果预测波动率超过 ATR 的一定比例，或者超过自定义阈值 (如 3%)
            # 说明此时是“乱战”，强制调低 gate_result.score 或关闭 allow
            if relative_vol > 0.03:  # 3% 的波动预期通常意味着高风险
                gate_result.allow = False
                signal_log(
                    f"⚠️ {ticker} 预测不确定性过高 ({relative_vol:.2%})，跳过开仓"
                )

        # 2. 基础计算
        predicted_up = calc(low, median, high, latest_price)
        slope = compute_hybrid_slope(median, close_df.values)     
        # 3. 初始信号判定
        raw_signal = (
            self._make_raw_signal(predicted_up, slope, model_score)
            if gate_result.allow
            else "HOLD"
        )
        # print(
        #     f"predicted_up: {predicted_up}, slope: {slope}, model_score: {model_score}, gate_result.allow: {gate_result.allow}"
        # )
        # 4. 账户状态同步 (Regime 合成)
        # 策略：只要账户风险层说是 bad，整体就是 bad；否则看信号动量

        is_trend_strong = slope > settings.SLOPE
        is_model_confident = model_score > settings.MODEL_TH
        is_gate_open = gate_result.allow

        is_vol_stable = True
        if np.std is not None:
            # 如果 std 在预测周期内剧烈发散，视为不稳定
            vol_growth = std[-1] / (std[0] + 1e-6)
            if vol_growth > 1.5:  # 预测未来波动会放大 50%
                is_vol_stable = False

        # 优化后的判定逻辑
        signal_regime = "neutral"

        # 1. 完美状态：全部达标
        if is_trend_strong and is_model_confident and is_gate_open and is_vol_stable:
            signal_regime = "good"
        # 增加一个特殊警告：如果斜率好但波动率失控
        elif is_trend_strong and not is_vol_stable:
            signal_regime = "neutral"  # 这种时候不能给 good，防止追高
            signal_log(f"⚠️ {ticker} 斜率强但预测波动率失控，保持中立")
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
        pos_dict = self.position_mgr.pos_to_dict(ticker)
        position_size = pos_dict.get("position_size", 0.0)
        has_position = position_size > 0

        liquidate_reason = None
        reduce_strength = eq_decision.reduce_strength  # 初始强度来自账户风控

        # 更新移动止损并检查
        self.position_mgr.update_trailing_stop(ticker, latest_price, atr)
        stop_loss_cooldown = 10
        if has_position:
            # A. 检查止损
            if latest_price <= pos_dict.get("stop_loss", 0):
                latest_price = pos_dict["stop_loss"] * 0.998  # 模拟滑点成交
                raw_signal = "LIQUIDATE"
                liquidate_reason = "STOP LOSS"
                self.position_mgr.cooldown[ticker] = stop_loss_cooldown

            # B. 检查账户风险指令 (REDUCE/LIQUIDATE)
            elif eq_decision.action in ("REDUCE", "LIQUIDATE"):
                raw_signal = eq_decision.action
                liquidate_reason = eq_decision.reason

            # C. 检查个股止盈
            tp_action = self.position_mgr.check_take_profit(ticker, latest_price)
            if isinstance(tp_action, float):
                raw_signal = "REDUCE"
                # 取账户减仓要求和个股止盈强度的最大值 (取严原则)
                reduce_strength = max(
                    float(reduce_strength or 0.0), float(tp_action or 0.0)
                )
                liquidate_reason = "TAKE_PROFIT"
                self.position_mgr.cooldown[ticker] = stop_loss_cooldown

        # 6. 计算最终缩放系数与分数
        final_gate_mult = gate_result.score * eq_decision.gate_mult
        raw_score = self._compute_debounce_score(
            raw_signal, model_score, final_gate_mult
        )

        # 7. 善后处理
        self.position_mgr.update_cooldown()
        # strength = compute_strength(slope=slope, gate=final_gate_mult)
        strength = compute_strength(
            slope=slope,
            gate=final_gate_mult,
            alpha=settings.STRENGTH_ALPHA,  # 确保是从动态配置对象里取的
            slope_threshold=settings.SLOPE,  # 同上
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
            position_size=position_size,
            raw_signal=raw_signal,
            raw_score=raw_score,
            reduce_strength=reduce_strength,
            liquidate_reason=liquidate_reason,
            strength=strength,  # 大于0才有开仓意图
            slope=slope,
        )

        # if raw_signal == "LONG":
        #     print(f"settings.STRENGTH_ALPHA: {settings.STRENGTH_ALPHA}")
        #     signal_log(ctx)
        if raw_signal == "LONG":
            signal_log(
                f"🔥 {ticker} slope={slope}|eq_decision.action={eq_decision.action} Lost_price={pos_dict.get('stop_loss', 0)}  raw_signal={raw_signal} | final_regime:{final_regime} | Price: {latest_price:.2f} | "
                f"predicted_up: {predicted_up:.3f} | Score: {model_score:.3f} | Gate_Mult: {final_gate_mult:.2f}"
            )
        # signal_log(ctx)
        return ctx
