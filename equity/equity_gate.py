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
