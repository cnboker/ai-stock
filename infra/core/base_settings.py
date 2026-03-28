# core/config.py

class BaseSettings:
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
