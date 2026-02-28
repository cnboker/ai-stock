from infra.core.context import TradingSession

class BacktestSession(TradingSession):
    def __init__(self, init_cash: float):
        super().__init__()
        self.position_mgr.reset(init_cash)