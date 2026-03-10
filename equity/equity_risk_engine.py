''' 
EquityRecorder     ← 记录资金曲线
      ↓
EquityFeature      ← 计算 dd / slope / zscore
      ↓
EquityRiskEngine   ← 最终交易决策
'''

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime
from strategy.trade_intent import TradeIntent


class EquityRiskEngine:

    def __init__(self, cooldown_mgr):
        self.cooldown_mgr = cooldown_mgr
        self.dd_level = 0

    def decide(self, eq_feat, has_position):

        regime = equity_regime(eq_feat)

        # 防抖
        regime = self.cooldown_mgr.update(regime)

        gate_mult = equity_gate(eq_feat)

        dd = eq_feat["eq_drawdown"].iloc[-1]
        slope = eq_feat["eq_slope"].iloc[-1]

        dd_abs = abs(dd)

        action = "HOLD"
        reduce_strength = 0.0
        reason = ""

        # ===== 最大回撤保护 =====

        if dd_abs > 0.10:
            action = "REDUCE"
            reduce_strength = 1.0
            reason = "max_drawdown"

        # ===== 中级回撤 =====

        elif dd_abs > 0.06:
            action = "REDUCE"
            reduce_strength = 0.6
            reason = "drawdown_mid"

        # ===== 小回撤 =====

        elif dd_abs > 0.03:
            action = "REDUCE"
            reduce_strength = 0.3
            reason = "drawdown_small"

        # ===== slope 崩坏 =====

        if slope < -0.002:
            action = "REDUCE"
            reduce_strength = max(reduce_strength, 0.5)
            reason = "equity_slope_break"

        # ===== bad regime =====

        if regime == "bad":
            gate_mult = 0.0

            if has_position:
                action = "REDUCE"
                reduce_strength = max(reduce_strength, 0.5)
                reason = "bad_regime"

        return TradeIntent(
            action=action,
            regime=regime,
            gate_mult=gate_mult,
            reduce_strength=reduce_strength,
            reason=reason,
        )