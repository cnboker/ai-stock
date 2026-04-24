import numpy as np
from infra.core.dynamic_settings import settings

def compute_strength(
    slope: float,
    gate: float,
    *,
    alpha: float = settings.STRENGTH_ALPHA,
    # 核心改动：直接使用入场拦截的那个阈值作为强度的起点
    slope_threshold: float = settings.SLOPE, 
):
    if gate < 0.2: return 0.0
    
    # 如果没过入场门槛，强度自然为 0
    if slope < slope_threshold: return 0.0

    # 强度从入场点开始线性增长
    # 这样当 slope 刚刚超过 threshold 时，强度从 0 附近平滑启动
    strength = gate * (slope - slope_threshold) * alpha
    
    return float(np.clip(strength, 0.0, 1.0))