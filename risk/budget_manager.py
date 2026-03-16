import numpy as np


class BudgetManager:
    """
    专业资金管理模块
    不改变 get_budget 接口
    """

    def __init__(
        self,
        max_signal_pct: float = 0.3,
        max_drawdown_pct: float = 0.2,
        single_position_limit: float = 0.2,
        risk_per_trade: float = 0.01,      # 每笔交易风险
        kelly_fraction: float = 0.5,       # Kelly缩放
        volatility_target: float = 0.02,   # 目标波动
    ):

        self.max_signal_pct = max_signal_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.single_position_limit = single_position_limit

        self.risk_per_trade = risk_per_trade
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target

        self._last_signal_score = {}

    def get_budget(
        self,
        ticker: str,
        gate_score: float,
        available_cash: float,
        equity: float,
        positions_value: float,
    ) -> float:

        # =============================
        # 1 信号平滑
        # =============================

        last_score = self._last_signal_score.get(ticker, gate_score)

        smooth_score = 0.7 * last_score + 0.3 * gate_score

        self._last_signal_score[ticker] = smooth_score

        # =============================
        # 2 信号资金限制
        # =============================

        signal_budget = available_cash * smooth_score * self.max_signal_pct

        # =============================
        # 3 账户回撤限制
        # =============================

        max_allowed = equity * (1 - self.max_drawdown_pct) - positions_value
        max_allowed = max(max_allowed, 0)

        # =============================
        # 4 单票仓位限制
        # =============================

        single_limit = equity * self.single_position_limit

        # =============================
        # 5 Kelly 资金
        # =============================

        kelly_budget = equity * self.kelly_fraction * smooth_score * 0.1

        # =============================
        # 6 每笔交易风险预算
        # =============================

        risk_budget = equity * self.risk_per_trade

        # =============================
        # 7 波动率目标仓位
        # =============================

        # 如果没有vol数据，就给一个保守值
        est_volatility = 0.02

        vol_budget = equity * (self.volatility_target / est_volatility)

        # =============================
        # 8 最终资金
        # =============================

        budget = min(
            signal_budget,
            max_allowed,
            single_limit,
            kelly_budget,
            vol_budget,
        )

        budget = min(budget, available_cash)

        return max(budget, 0)


budget_mgr = BudgetManager(
    max_signal_pct=0.3,
    max_drawdown_pct=0.2,
    single_position_limit=0.2,
)