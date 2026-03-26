import numpy as np
import logging
from dataclasses import dataclass
from infra.core.config import settings

# 配置日志，方便观察资金被哪个维度拦截
logger = logging.getLogger(__name__)

class BudgetManager:
    """
    专业资金管理模块 (优化版)
    解决多重 min() 导致的资金过低问题，增强高分信号的可用性
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.2,       # 最大账户回撤保护阈值
        single_position_limit: float = 0.3, # 单票仓位上限 (建议调高到 0.3, 增加进攻性)
        risk_per_trade: float = 0.02,       # 每笔交易愿意承担的账户风险
        volatility_target: float = 0.02,    # 目标波动率
        min_trade_value: float = 3000.0     # 最低起购金额 (低于此值不交易，防止手续费吞噬利润)
    ):
        # 从配置中读取，若无则使用默认
     
        self.kelly_fraction = getattr(settings, 'KELLY_FRACTION', 0.5)
        
        self.max_drawdown_pct = max_drawdown_pct
        self.single_position_limit = single_position_limit
        self.risk_per_trade = risk_per_trade
        self.volatility_target = volatility_target
        self.min_trade_value = min_trade_value

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
        核心资金分配逻辑
        """
        if gate_score <= 0:
            return 0.0

        # =============================
        # 1. 信号平滑 (防止单根K线分数剧变导致频繁调仓)
        # =============================
        last_score = self._last_signal_score.get(ticker, gate_score)
        smooth_score = 0.6 * last_score + 0.4 * gate_score
        self._last_signal_score[ticker] = smooth_score

        # =============================
        # 2. 基于信号强度的预算 (核心改进)
        # 不再使用 available_cash，改用 equity 配合分数，防止资金利用率过低
        # =============================
        # 信号越接近 1.0，预算越接近我们允许的最大单票限制
        signal_limit = equity * self.single_position_limit
        signal_budget = signal_limit * (smooth_score / 1.0) 

        # =============================
        # 3. 账户回撤硬限制 (风控底线)
        # 如果你当前的 positions_value（持仓市值）已经接近或超过了你设定的 equity * (1 - max_drawdown_pct)，那么 max_allowed 就会直接变成 0。
        # =============================
        max_allowed = equity * (1 - self.max_drawdown_pct) - positions_value
        max_allowed = max(max_allowed, 0)

        # =============================
        # 4. 单票硬上限
        # =============================
        single_limit = equity * self.single_position_limit

        # =============================
        # 5. 改进版 Kelly 资金
        # =============================
        kelly_budget = equity * self.kelly_fraction * smooth_score 

        # =============================
        # 6. 动态每笔风险预算 (核心改进)
        # =============================
        # 基础风险 2%，但如果分数极高 (>0.75)，允许风险额翻倍到 4%
        dynamic_risk_ratio = self.risk_per_trade
        if smooth_score > 0.8:
            dynamic_risk_ratio *= 2.0  # 给予超级信号双倍信任
        elif smooth_score > 0.7:
            dynamic_risk_ratio *= 1.5
            
  
        risk_budget = equity * dynamic_risk_ratio

        # =============================
        # 7. 最终资金决策 (取所有约束的最小值)
        # =============================
        # 剔除了不稳定的 vol_budget，保留核心风控项
        raw_budget = min(
            signal_budget,
            max_allowed,
            single_limit,
            kelly_budget,
            risk_budget
        )

        # 确保不超过当前可用现金
        final_budget = min(raw_budget, available_cash)
        print(f'final_budget->${final_budget}')
        # =============================
        # 8. 可用性保底逻辑 (重要！)
        # =============================
        # 如果计算出的资金太少，且我们还有钱，且信号足够强(>0.65)，强制给一个起步价
        if final_budget < self.min_trade_value:
            if smooth_score > 0.65 and available_cash >= self.min_trade_value:
                # 高分信号，给予保底仓位入场
                final_budget = self.min_trade_value
            else:
                # 信号一般且钱不够买一手，放弃，避免碎单
                return 0.0

        # 日志监控：当你发现买不进去时，看看到底是哪个项在起作用
        # logger.debug(
        #     f"[{ticker}] Score:{smooth_score:.2f} | "
        #     f"Budgets -> Signal:{signal_budget:.0f}, MaxAllow:{max_allowed:.0f}, "
        #     f"Kelly:{kelly_budget:.0f}, Risk:{risk_budget:.0f} | Final:{final_budget:.0f}"
        # )

        return float(max(final_budget, 0))

# ===================== 模块级单例初始化 =====================
# 默认给予 20% 的单票上限，保证 10 万资金能买入 2 万左右的仓位
budget_mgr = BudgetManager(
    max_drawdown_pct=0.25,      # 稍微放宽回撤保护，给策略呼吸空间
    single_position_limit=0.3,   # 单票上限 30%
    risk_per_trade=0.02,         # 每笔交易愿意亏账户的 2%
    min_trade_value=3000.0       # 低于 3000 元不玩
)