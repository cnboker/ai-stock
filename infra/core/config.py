# core/config.py

class GlobalConfig:
    # --- 入场控制 ---
    MODEL_LONG_THRESHOLD = 0.5
    TREND_SLOPE_THRESHOLD = 0
    PREDICTED_UP = -0.05
    # --- 止盈止损阶段触发 ---
    INIT_PROFIT_TRIGGER = 0.05
    TREND_STAGE_TRIGGER = 0.12
    
    # --- 移动止损参数 ---
    ATR_MULTIPLIER = 1.5
    TP1_RATIO = 1.08
    TP2_RATIO = 1.15
    
    # --- 资金管理 ---
    KELLY_FRACTION = 0.5
    # --- 单笔最大仓位 ---
    MAX_SIGNAL_PCT = 0.3

    # --- 风控
    RISK_PER_TRADE = 0.02 # 单笔最多亏 1%
    ATR_STOP_MULT = 1.5
    ATR_TAKE_MULT = 4.0  # ATR 止盈
    MAX_STOP_PCT = 0.05
    MIN_STOP_PCT = 0.01

    MIN_RR = 0.9 #默认最小盈亏比 1.5
    
# 实例化为一个全局单例
settings = GlobalConfig()