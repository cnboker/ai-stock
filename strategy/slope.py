import numpy as np

def compute_slope(prices, window=20):
   
    if len(prices) < window:
        return 0.0

    y = prices[-window:]
    x = np.arange(window)

    slope = np.polyfit(x, y, 1)[0]

    # 归一化
    slope = slope / np.mean(y)

    return float(slope)

def corrected_slope(
    slope_raw: float,
    prices: np.ndarray,
    k: int = 5,
    alpha: float = 0.7,
    scale: float = 3.0,
):
    """
    slope_raw : 原始 slope
    prices    : 最近价格
    k         : 短期窗口
    alpha     : slope 权重
    scale     : 去饱和尺度
    """

    # --- 去饱和 ---
    slope_norm = slope_raw / (abs(slope_raw) + scale)

    if len(prices) < k + 1:
        return slope_norm

    # --- 短期动量 ---
    recent_ret = (prices[-1] - prices[-k]) / prices[-k]

    # 轻度压缩
    recent_dir = np.tanh(recent_ret * 3)

    # --- 融合 ---
    slope_fixed = alpha * slope_norm + (1 - alpha) * recent_dir

    return float(np.clip(slope_fixed, -1.0, 1.0))