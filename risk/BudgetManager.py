import numpy as np

class BudgetManager:
    """
    将 Gate score 转换为实际信号资金预算。
    支持：
        - 信号强度平滑化
        - 最大账户回撤限制
        - 单票最大占用比例
    """

    def __init__(
        self,
        max_signal_pct: float = 0.3,  # gate_score=1 最大动用占总现金比例
        max_drawdown_pct: float = 0.2,  # 账户最大可回撤
        single_position_limit: float = 0.1,  # 单票最大占比
    ):
        self.max_signal_pct = max_signal_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.single_position_limit = single_position_limit

        # 历史信号平滑
        self._last_signal_score = {}

    def get_budget(
        self,
        ticker: str,
        gate_score: float,
        available_cash: float,
        equity: float,
        positions_value: float,
    ) -> float:
      
        """
        返回本次信号可用资金
        """
        # ---------------- 平滑化处理 ----------------
        last_score = self._last_signal_score.get(ticker, gate_score)
        smooth_score = 0.7 * last_score + 0.3 * gate_score
        self._last_signal_score[ticker] = smooth_score

        # ---------------- 基于账户资金 ----------------
        max_signal_cap = available_cash * smooth_score * self.max_signal_pct

        # ---------------- 基于回撤限制 ----------------
        max_allowed = equity * (1 - self.max_drawdown_pct) - positions_value
        max_allowed = max(max_allowed, 0)
        max_signal_cap = min(max_signal_cap, max_allowed)

        # ---------------- 单票限制 ----------------
        max_signal_cap = min(max_signal_cap, equity * self.single_position_limit)

        return max_signal_cap


budget_mgr = BudgetManager(
    max_signal_pct=0.3,
    max_drawdown_pct=0.2,
    single_position_limit=0.1,
)