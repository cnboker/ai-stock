from typing import Optional
from strategy.regime_cooldown import regime_cooldown
from strategy.equity_policy import decide_equity_policy


class EquityEngine:
    """
    独立的资金/策略决策引擎
    """

    def __init__(self, cooldown_mgr=None):
        self.cooldown_mgr = cooldown_mgr or regime_cooldown

    def decide(self, eq_feat, has_position, ticker: Optional[str] = None):
        """
        返回 eq_decision 和 raw_action
        """
        eq_decision = decide_equity_policy(eq_feat, has_position)
        '''
        防抖
        bad 进得快，出得慢
        good 要连续确认
        neutral 是缓冲态
        '''
        if ticker:
            eq_decision.regime = self.cooldown_mgr.update(ticker, eq_decision.regime)

        raw_action = eq_decision.action
        if eq_decision.regime == "bad" and not raw_action:
            raw_action = "HOLD"

        return eq_decision, raw_action
