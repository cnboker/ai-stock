import numpy as np


def compute_slope(
    predicted_up: float,
    *,
    horizon: int = 1,
    slope_scale: float = 0.01,
):
    """
    slope ∈ [-1, 1]
    predicted_up: 预测涨跌幅（如 +0.005 = +0.5%）
    horizon: 预测跨度（bar 数）
    """

    if predicted_up is None:
        return 0.0

    raw_slope = predicted_up / max(1, horizon)

    # tanh 压缩，避免极端值
    slope = np.tanh(raw_slope / slope_scale)

    return float(slope)


def corrected_slope(
    slope_raw: float,
    prices: np.ndarray,
    k: int = 5,
    alpha: float = 0.7,
):
    """
    slope_raw: 你现在算出来的 slope
    prices: 最新价格数组（tick 或 close）
    k: 用最近 k 个点
    alpha: 原 slope 保留权重
    """

    if len(prices) < k + 1:
        return slope_raw

    recent_ret = (prices[-1] - prices[-k]) / prices[-k]

    # 映射到 slope 量纲（避免过大）
    recent_dir = np.tanh(recent_ret * 50)

    slope_fixed = alpha * slope_raw + (1 - alpha) * recent_dir
    return slope_fixed
