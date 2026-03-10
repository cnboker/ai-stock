import numpy as np


def compute_strength(
    slope: float,
    gate: float,
    *,
    alpha: float = 1.5,
    max_strength: float = 1.0,
):

    if gate < 0.2:
        return 0.0

    # 只允许上涨
    slope_pos = max(0.0, slope)

    if slope_pos < 0.05:
        return 0.0

    slope_strength = slope_pos ** alpha

    strength = gate * slope_strength

    return float(np.clip(strength, 0.0, max_strength))