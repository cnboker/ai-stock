from typing import Optional
import numpy as np

from equity.equity_gate import equity_gate
from equity.equity_regime import equity_regime


RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
"""
价格预测（Chronos）
        ↓
结构判断（gater）
        ↓
资金状态审查（equity_regime）
        ↓
资金状态放大/压制（equity_gate）
        ↓
信号稳定器（debouncer）

"""


class SignalManager:
    def __init__(
        self,
        gater,
        debouncer_manager,
        min_score: float = 0.05,
    ):
        self.gater = gater
        self.debouncer = debouncer_manager
        self.min_score = min_score

    def evaluate(
        self,
        *,
        ticker: str,
        low: float,
        median: float,
        high: float,
        latest_price: float,
        context,
        model_score: float,
        eq_feat,
        has_position: bool,
        atr: float = 1.0,
    ) -> tuple[str, float, any]:
        """
        返回最终交易信号:
        LONG / SHORT / HOLD / None
        """

        # =========================================================
        # 1️⃣ 原始 Gate（价格 + 预测结构）
        # =========================================================
        gate_result = self.gater.evaluate(
            lower=low,
            mid=median,
            upper=high,
            context=context.values,
        )

        if not gate_result.allow:
            raw_signal = "HOLD"
        else:
            raw_signal = make_signal(
                low=low,
                median=median,
                high=high,
                latest_price=latest_price,
            )
        print("gate_result", gate_result)
        # =========================================================
        # 2️⃣ Equity Regime（硬熔断）
        # =========================================================
        if eq_feat is None or eq_feat.empty:
            regime = "neutral"  # 默认倍率，或根据业务逻辑
        else:
            regime = equity_regime(eq_feat)
        if regime == "bad":
            if has_position:
                raw_signal = "REDUCE"
            else:
                raw_signal = "HOLD"
        print("regime", regime)
        # =========================================================
        # 3️⃣ Equity Gate（软放大 / 压制）
        # =========================================================
       
        if eq_feat is None or eq_feat.empty:
            gate_mult = 1.0  # 默认倍率，或根据业务逻辑
        else:
            gate_mult = equity_gate(eq_feat)
        print("gate_mult", gate_mult)
        # ===== final_score =====
        if raw_signal == "LONG":
            final_score = +model_score * gate_mult
        elif raw_signal == "SHORT":
            final_score = -model_score * gate_mult
        elif raw_signal == "REDUCE":
            """
             回撤	减仓力度
            -2%	小幅减
            -5%	明显减
            -8%	强制减
            """
            dd = abs(eq_feat["eq_drawdown"].iloc[-1])
            reduce_strength = min(1.0, dd / 0.08)  # 8% 回撤 → 满减仓
            final_score = -reduce_strength * gate_mult
        else:
            final_score = 0.0
        # =========================================================
        # 4️⃣ 弱信号过滤（避免抖动）,避免min_score 会“吃掉 REDUCE”
        # =========================================================
        if raw_signal != "REDUCE" and abs(final_score) < self.min_score:
            final_score = 0.0

        # =========================================================
        # 5️⃣ Debounce（最终裁决）
        # =========================================================
        """
        raw_signal	含义
        > 0	LONG（强度 = 数值）
        < 0	SHORT / REDUCE
        = 0	HOLD
        """
        final_action, confidence = self.debouncer.update(ticker, final_score, atr=atr)

        return final_action, confidence, gate_result


def make_signal(
    low: np.ndarray,
    median: np.ndarray,
    high: np.ndarray,
    latest_price: float,
    up_thresh: float = 0.006,  # +0.6%
    down_thresh: float = -0.006,  # -0.6%
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
    up_ratio = (pred_high - latest_price) / latest_price
    down_ratio = (pred_low - latest_price) / latest_price
    mid_ratio = (pred_mid - latest_price) / latest_price

    # ======== 多条件过滤，防止假信号 ========

    # 做多条件：
    if mid_ratio > up_thresh and down_ratio > -0.003:  # 下方风险受控
        return "LONG"

    # 做空条件：
    if mid_ratio < down_thresh and up_ratio < 0.003:  # 上方风险受控
        return "SHORT"

    return "HOLD"


def print_signal(ticker, signal):
    if signal == "LONG":
        print(f"{RED}{ticker} 实盘信号: {signal}{RESET}")
    elif signal == "SHORT":
        print(f"{GREEN}{ticker} 实盘信号: {signal}{RESET}")
    elif signal == "REDUCE":
        print(f"{ticker} 实盘信号: ⚠️ REDUCE")

    else:
        print(f"{ticker} 实盘信号: {signal}")
