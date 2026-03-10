from equity.equity_risk_engine import EquityRiskEngine
from equity.regime_cooldown import regime_cooldown
from strategy.trade_intent import TradeIntent

class EquityState:
    def __init__(self):
        self.dd_level = 0

'''
EquityRecorder
      ↓
EquityFeature
      ↓
EquityRegime
      ↓
EquityGate
      ↓
EquityRiskEngine
      ↓
TradeIntent
'''
class EquityEngine:

    def __init__(self):
        
        self.risk_engine = EquityRiskEngine(regime_cooldown)

    def decide(self, eq_feat, has_position):
        if eq_feat is None or eq_feat.empty:
            return TradeIntent(
                action="HOLD",
                regime="neutral",
                gate_mult=1.0,
            )
        decision = self.risk_engine.decide(eq_feat, has_position)

        return decision