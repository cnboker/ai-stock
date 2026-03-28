# core/config.py

class BaseSettings:
    # --- 入场控制 ---
    MODEL_TH = 0.46
    SLOPE = 0.0001
    PREDICT_UP = 0
    # --- 止盈止损阶段触发 ---
    INIT_PT = 0.07
    TREND_STAGE = 0.18
    
    # --- 移动止损参数 ---
    ATR_MULT = 3.87
    TP1 = 1.02
    TP2 = 1.19
    
    # --- 资金管理 ---
    KELLY = 0.3
 

    # --- 风控
    RISK = 0.015 # 单笔最多亏 1%
    ATR_STOP = 3.4
    MAX_STOP = 0.08
    MIN_STOP = 0.01

   
    STRENGTH_ALPHA = 1.34
    SLOPE_MIN = 0.02
    CONFIRM_N = 3
