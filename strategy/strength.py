import numpy as np
from infra.core.dynamic_settings import settings


def compute_strength(
    slope: float,
    gate: float,
    *,
    alpha: float = settings.STRENGTH_ALPHA,       # 建议改为 5.0 ~ 20.0 (线性系数)
    slope_min: float = settings.SLOPE_MIN or 0.001, # 门槛必须降下来
):
    if gate < 0.2: return 0.0
    
    # 1. 降低门槛：0.02 太高了，改成 0.001 甚至更小
    if slope < slope_min: return 0.0

    # 2. 线性映射代替指数：(slope - min) * alpha
    # 这样 0.01 的斜率在 alpha=20 时能得到 0.2 的强度
    strength = gate * (slope - slope_min) * alpha
    
    return float(np.clip(strength, 0.0, 1.0))