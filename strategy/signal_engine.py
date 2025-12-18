import numpy as np
# signal/signal_engine.py
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def make_signal(
    low: np.ndarray,
    median: np.ndarray,
    high: np.ndarray,
    last_price: float,
    up_thresh: float = 0.006,    # +0.6%
    down_thresh: float = -0.006, # -0.6%
):
    """
    根据 Chronos 预测生成交易信号
    返回: "LONG" | "SHORT" | "HOLD"
    """

    if len(median) == 0:
        return "HOLD"

    # 使用最后一步预测
    pred_mid = median[-1]
    pred_low = low[-1]
    pred_high = high[-1]

    # 相对涨跌
    up_ratio = (pred_high - last_price) / last_price
    down_ratio = (pred_low - last_price) / last_price
    mid_ratio = (pred_mid - last_price) / last_price

    # ======== 多条件过滤，防止假信号 ========

    # 做多条件：
    if (
        mid_ratio > up_thresh
        and down_ratio > -0.003     # 下方风险受控
    ):
        return "LONG"

    # 做空条件：
    if (
        mid_ratio < down_thresh
        and up_ratio < 0.003        # 上方风险受控
    ):
        return "SHORT"

    return "HOLD"

def print_signal(ticker, signal):
    if signal == "LONG":
        print(f"{RED}{ticker} 实盘信号: {signal}{RESET}")
    elif signal == "SHORT":
        print(f"{GREEN}{ticker} 实盘信号: {signal}{RESET}")
    else:
        print(f"{ticker} 实盘信号: {signal}")
