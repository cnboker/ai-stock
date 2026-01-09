import numpy as np


def compute_strength(
    slope: float,
    gate: float,
    *,
    alpha: float = 1.5,
    max_strength: float = 1.0,
):
    """
    strength ∈ [0, max_strength]
    """

    # 没方向 or gate 太小
    if abs(slope) < 0.05 or gate < 0.2:
        return 0.0

    # slope → 非线性强度
    slope_strength = abs(slope) ** alpha

    strength = gate * slope_strength

    return float(np.clip(strength, 0.0, max_strength))
