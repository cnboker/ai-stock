''' 
EquityRecorder     ← 记录资金曲线
      ↓
EquityFeature      ← 计算 dd / slope / zscore
      ↓
EquityRiskEngine   ← 最终交易决策

DD > -3%      risk = 1.0
-3% ~ -6%     risk = 0.8
-6% ~ -10%    risk = 0.6
-10% ~ -15%   risk = 0.4
< -15%        risk = 0.25
不会卡死系统
'''

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime
from strategy.trade_intent import TradeIntent


class EquityRiskEngine:

    def __init__(self, cooldown_mgr):
        self.cooldown_mgr = cooldown_mgr

    def compute_risk_scale(self, dd):

        dd_abs = abs(dd)

        if dd_abs < 0.03:
            return 1.0

        if dd_abs < 0.06:
            return 0.8

        if dd_abs < 0.10:
            return 0.6

        if dd_abs < 0.15:
            return 0.4

        return 0.25

    def decide(self, eq_feat, has_position):

        regime = equity_regime(eq_feat)

        # 防抖
        regime = self.cooldown_mgr.update(regime)

        gate_mult = equity_gate(eq_feat)

        dd = eq_feat["eq_drawdown"].iloc[-1]
        slope = eq_feat["eq_slope"].iloc[-1]

        action = "HOLD"
        reduce_strength = 0.0
        reason = ""

        # ======================
        # 1️⃣ 风险缩放
        # ======================

        risk_scale = self.compute_risk_scale(dd)

        gate_mult *= risk_scale

        # ======================
        # 2️⃣ slope 崩坏保护
        # ======================

        if slope < -0.002:

            if has_position:
                action = "REDUCE"
                reduce_strength = 0.5

            gate_mult *= 0.7

            reason = "equity_slope_break"

        # ======================
        # 3️⃣ regime 风控
        # ======================

        if regime == "bad":

            # 不完全锁死
            gate_mult *= 0.3

            if has_position:
                action = "REDUCE"
                reduce_strength = max(reduce_strength, 0.5)

            reason = "bad_regime"

        # ======================
        # 4️⃣ 极端回撤保护
        # ======================

        if abs(dd) > 0.18:

            gate_mult = 0

            if has_position:
                action = "REDUCE"
                reduce_strength = 1.0

            reason = "max_drawdown"

        return TradeIntent(
            action=action,
            regime=regime,
            gate_mult=gate_mult,
            reduce_strength=reduce_strength,
            reason=reason,
        )