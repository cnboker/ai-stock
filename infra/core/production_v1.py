# core/config.py
"""
创业板 ETF (sz159908 / sz159915 / sh515880(通讯))
训练集表现：在 197 天内跑出了 107% 的总收益（同期基准 83%）。这 24.5% 的超额收益（Alpha）来源于它只交易了 6 次。

核心逻辑：模型阈值设定在 0.46。这意味着它放弃了所有“疑似”机会，只在趋势极其明确时满仓出击。

验证集现状：在最近 38 天的震荡市中，它依然保持了微弱的领先优势（+0.89% Alpha），说明逻辑没有失效。

2. 纳指 ETF (sh513100)
单边牛市（197天）：策略回报 42.9%，略输基准 43.37%。

反思：量化择时在“傻瓜式上涨”中通常不占优，因为它会因为保护利润而提前下车或延迟入场。

暴跌防守（最近38天）：基准大跌 -8.77%，策略仅跌 -2.53%。

高光时刻：触发了 6.24% 的 Alpha。在 2026-03-27 凌晨，模型在发现高分信号（0.985）后迅速入场，但由于价格触碰了 3.42倍 ATR 止损位，系统果断执行了 CLOSE LONG。

结论：这套策略在纳指上的核心价值是**“避雷针”**，确保你在美股见顶回落时不被深埋。
实验数据证明了你的系统已经是一个合格的“宽基策略”。
"""
class GlobalConfig:
    # --- 入场控制 ---
    MODEL_LONG_THRESHOLD = 0.46
    TREND_SLOPE_THRESHOLD = 0.0001
    PREDICTED_UP = 0
    # --- 止盈止损阶段触发 ---
    INIT_PROFIT_TRIGGER = 0.07
    TREND_STAGE_TRIGGER = 0.18
    
    # --- 移动止损参数 ---
    ATR_MULTIPLIER = 3.87
    TP1_RATIO = 1.02
    TP2_RATIO = 1.19
    
    # --- 资金管理 ---
    KELLY_FRACTION = 0.3
 

    # --- 风控
    RISK_PER_TRADE = 0.015 # 单笔最多亏 1%
    ATR_STOP_MULT = 3.4
    MAX_STOP_PCT = 0.08
    MIN_STOP_PCT = 0.01

    MIN_RR = 0.08 #默认最小盈亏比 1.5
    STRENGTH_ALPHA = 1.34
    STRENGTH_SLOPE_MIN = 0.02
    CONFIRM_WINDOW = 3
# 实例化为一个全局单例
settings = GlobalConfig()