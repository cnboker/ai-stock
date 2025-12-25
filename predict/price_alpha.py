import numpy as np

#放大值信区
def chronos2_to_large_style(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    context: np.ndarray,
    base_alpha: float = 1.4,
    vol_window: int = 32,
    max_alpha: float = 2.5,
):
    """
    Chronos-2 → Chronos-large 风格区间后处理

    参数
    ----
    q10, q50, q90 : Chronos-2 输出分位
    context       : 历史真实序列（1D）
    base_alpha    : 基础放大倍率（1.3~1.6 推荐）
    vol_window    : 计算历史波动的窗口
    max_alpha     : 最大放大上限（防炸）

    返回
    ----
    lower, mid, upper
    """

    # ---------- 1. 模型给的区间 ----------
    model_spread = q90 - q10
    model_spread = np.maximum(model_spread, 1e-6)

    # ---------- 2. 历史真实波动 ----------
    recent = context[-vol_window:]
    realized_vol = np.std(recent)

    # ---------- 3. 动态放大因子 ----------
    alpha = base_alpha * realized_vol / (np.mean(model_spread) + 1e-6)
    alpha = np.clip(alpha, 1.0, max_alpha)

    # ---------- 4. 只放大区间，不动中位数 ----------
    half = model_spread / 2
    lower = q50 - alpha * half
    upper = q50 + alpha * half

    return lower, q50, upper
