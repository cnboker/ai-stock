import numpy as np

def compute_hybrid_slope(median_prices, hist_close_prices):
    hist_std = hist_close_prices.std()

    # 1. 预测斜率 (当前为 0.0)
    slope_pred = __compute_refined_slope_for_prediction(median_prices, hist_std)
    
    # 2. 历史斜率 (当前表现活跃: -0.09 ~ 0.15)
    slope_hist = __compute_refined_slope(hist_close_prices)

    # 3. 动态融合：
    # 如果预测失效(diff=0)，则 100% 信任历史趋势
    # 如果预测有效，则各占 50%
    pred_diff = np.abs(median_prices[-1] - median_prices[0])
    if pred_diff < 1e-7:
        return slope_hist 
    else:
        return (slope_hist * 0.5) + (slope_pred * 0.5)
    

def __compute_refined_slope_for_prediction(median_prices, hist_std):
    
    """
    median_prices: 模型预测的未来价格中位数序列
    hist_std: 该标的历史真实价格的波动标准差 (从 df_context 获得)
    """
    y = np.array(median_prices)
    n = len(y)
    x = np.arange(n)
    
    # 1. 依然使用最小二乘法求出预测的价格变化速率 (Raw Slope)
    numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = n * np.sum(x**2) - (np.sum(x)**2)
    raw_slope = numerator / denominator if denominator != 0 else 0.0
    
    # 2. 归一化：使用历史波动的标准差
    # 含义：模型预测的每步涨幅，相当于历史波动水平的多少倍？
    norm_slope = raw_slope / (hist_std + 1e-9)
    
    return float(norm_slope)

def __compute_refined_slope(prices, window=20):
    if len(prices) < window:
        return 0.0
    
    y = np.array(prices[-window:])
    x = np.arange(window)
    
    # 1. 快速最小二乘法计算
    n = window
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x**2
    
    raw_slope = numerator / denominator if denominator != 0 else 0.0
    
    # 2. 顶级归一化：结合波动率 (使用标准差)
    # 这样得到的斜率代表：价格变动方向偏离了多少个标准差
    std_dev = np.std(y)
    if std_dev == 0: return 0.0
    
    # 归一化后的斜率，更具普适性
    norm_slope = raw_slope / std_dev
    
    return float(norm_slope)

