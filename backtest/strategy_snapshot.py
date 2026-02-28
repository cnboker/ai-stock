# backtest/strategy_snapshot.py

class StrategySnapshot:
    """
    记录一次回测所使用的“完整策略状态”
    """
    def __init__(self):
        from strategy.gate import gater
        from strategy.signal_mgr import SignalManager
        from risk.budget_manager import budget_mgr

        self.gate_threshold = gater.threshold
        self.gate_decay = gater.decay
        self.min_score = SignalManager.min_score
        self.budget_curve_k = budget_mgr.curve_k

    def restore(self):
        from strategy.gate import gater
        from strategy.signal_mgr import SignalManager
        from risk.budget_manager import budget_mgr

        gater.threshold = self.gate_threshold
        gater.decay = self.gate_decay
        SignalManager.min_score = self.min_score
        budget_mgr.curve_k = self.budget_curve_k