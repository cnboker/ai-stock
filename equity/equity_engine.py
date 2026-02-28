import numpy as np
from log import signal_log
from typing import Optional
from strategy.regime_cooldown import regime_cooldown
from strategy.equity_policy import decide_equity_policy


class EquityState:
    def __init__(self):
        self.dd_level = 0


class EquityEngine:
    """
    独立的资金/策略决策引擎
    """

    def __init__(self, cooldown_mgr=None):
        self.cooldown_mgr = cooldown_mgr or regime_cooldown
        self.state = EquityState()

    def decide(self, eq_feat, has_position):
        """
        返回 eq_decision 和 raw_action
        """
        eq_decision = decide_equity_policy(eq_feat, has_position, self.state)
        """
        防抖
        bad 进得快，出得慢
        good 要连续确认
        neutral 是缓冲态
        """
        eq_decision.regime = self.cooldown_mgr.update(new_regime=eq_decision.regime)

        raw_action = eq_decision.action

        if eq_decision.regime == "bad" and not raw_action:
            action = "HOLD"
        else:
            action = raw_action

        eq_decision.raw_action = raw_action
        eq_decision.action = action

        return eq_decision

    def log_equity_decision(self, eq_feat, decision):
        if eq_feat is None or eq_feat.empty:
            return

        dd = eq_feat["eq_drawdown"].iloc[-1]
        slope = eq_feat["eq_slope"].iloc[-1]

        signal_log(
            f"regime={decision.regime} "
            f"dd={dd:.2%} "
            f"slope={slope:.4f} "
            f"={decision.gate_mult:.2f} "
            f"action={decision.action} "
            f"strength={decision.reduce_strength:.2f}"
        )

    def calc_predicted_up_risk_adjusted(
        self,
        low: np.ndarray,
        median: np.ndarray,
        high: np.ndarray,
        latest_price: float,
    ):
        if latest_price <= 0:
            return 0.0

        up = (median[-1] - latest_price) / latest_price
        risk = (median[-1] - low[-1]) / latest_price

        return float(up - 0.5 * risk)
