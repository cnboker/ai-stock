import numpy as np
from infra.core.dynamic_settings import settings


def compute_strength(
    slope: float,
    gate: float,
    *,
    alpha: float = settings.STRENGTH_ALPHA,       # 建议 1.2 ~ 1.5
    slope_min: float = settings.SLOPE_MIN or 0.02, # 建议 0.02
):
    if gate < 0.2: return 0.0
    
    slope_pos = max(0.0, slope)
    if slope_pos < slope_min: return 0.0

    # 也可以考虑对 gate 也做一次非线性增强
    strength = gate * (slope_pos ** alpha)
    return float(np.clip(strength, 0.0, 1.0))