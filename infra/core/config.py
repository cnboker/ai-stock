# core/config.py

class GlobalConfig:
    # --- 入场控制 ---
    MODEL_LONG_THRESHOLD = 0.5
    TREND_SLOPE_THRESHOLD = -0.01
    PREDICTED_UP = -0.03
    # --- 止盈止损阶段触发 ---
    INIT_PROFIT_TRIGGER = 0.07
    TREND_STAGE_TRIGGER = 0.24
    
    # --- 移动止损参数 ---
    ATR_MULTIPLIER = 3.0
    TP1_RATIO = 1.1
    TP2_RATIO = 1.20
    
    # --- 资金管理 ---
    KELLY_FRACTION = 0.11
 

    # --- 风控
    RISK_PER_TRADE = 0.03 # 单笔最多亏 1%
    ATR_STOP_MULT = 0.56
    MAX_STOP_PCT = 0.08
    MIN_STOP_PCT = 0.01

    MIN_RR = 0.7 #默认最小盈亏比 1.5
    
# 实例化为一个全局单例
settings = GlobalConfig()