import numpy as np

def calculate_hurst_simple(ts, max_lag=20):
    """
    计算简易 Hurst 指数，用于辅助判断市场状态
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def risk_adjusted_decision(chronos_signal, price_history, threshold_low=0.45, threshold_high=0.55):
    """
    基于 Hurst 指数的决策过滤器
    chronos_signal: Chronos-Bolt 输出的预测值 (1 为看多, -1 为看空, 0 为观望)
    price_history: 过去一段时期的收盘价序列
    """
    h_index = calculate_hurst_simple(price_history)
    
    # 逻辑 1：市场处于高度随机/均值回归状态 (最近的拉垮行情特征)
    if h_index < threshold_low:
        # 强制观望或极轻仓，忽略 Chronos 的激进信号
        final_action = 0 
        reason = f"Hurst({h_index:.2f}) < {threshold_low}: 市场处于非趋势震荡，屏蔽信号"
        
    # 逻辑 2：市场展现出明显的趋势持续性
    elif h_index > threshold_high:
        final_action = chronos_signal
        reason = f"Hurst({h_index:.2f}) > {threshold_high}: 趋势确认，执行 Chronos 信号"
        
    # 逻辑 3：中间地带，减仓操作
    else:
        final_action = chronos_signal * 0.5 
        reason = f"Hurst({h_index:.2f}) 处于中性区间，执行 50% 仓位限制"
        
    return final_action, h_index, reason

# 示例调用
# signal = model_chronos.predict(data) 
# action, h, msg = risk_adjusted_decision(signal, recent_prices)