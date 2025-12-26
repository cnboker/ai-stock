from position.gate_reject_policy import PositionAction


class PositionExecutor:
    def __init__(self, position_mgr):
        self.position_mgr = position_mgr

    def execute(self, action: PositionAction, symbol: str):
        self.position_mgr.apply_action(symbol, action)
