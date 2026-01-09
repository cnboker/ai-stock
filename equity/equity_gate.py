import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Equity Gate（权重放大器，核心）
def equity_gate(eq_feat):
    """
    输出 [0, 1.5] 的 gate
    """
    ret_z = eq_feat["eq_ret_z"].iloc[-1]
    dd = eq_feat["eq_drawdown"].iloc[-1]

    score = (
        2.0 * ret_z
        - 3.0 * abs(dd)
    )

    return 0.5 + sigmoid(score)


def compute_gate(
    eq_feat,
    *,
    max_gate: float = 1.2,
    min_gate: float = 0.0,
):
    """
    gate ∈ [0, max_gate]
    决定：是否允许交易 + 风险强度
    """

    # ===== 兜底 =====
    if eq_feat is None or eq_feat.empty:
        return 1.0

    # ===== 取最新特征 =====
    dd = float(eq_feat["eq_drawdown"].iloc[-1])      # 负数
    atr = float(eq_feat.get("atr", 0.0).iloc[-1])
    regime = eq_feat.get("regime", "neutral")

    gate = 1.0

    # ===== 1. 回撤压制（最重要）=====
    if dd < -0.12:
        gate *= 0.25
    elif dd < -0.08:
        gate *= 0.45
    elif dd < -0.05:
        gate *= 0.70

    # ===== 2. 波动率压制 =====
    # 假设 atr 已经是归一化或相对值
    if atr > 2.0:
        gate *= 0.7
    elif atr < 0.6:
        gate *= 1.05  # 低波动轻微放行

    # ===== 3. Regime 调制 =====
    if regime in ("panic", "crash"):
        gate *= 0.3
    elif regime in ("bear",):
        gate *= 0.6
    elif regime in ("bull",):
        gate *= 1.05

    return float(np.clip(gate, min_gate, max_gate))
