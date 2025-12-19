from dataclasses import dataclass
from typing import Optional


@dataclass
class TradePlan:
    allow_trade: bool
    reason: str = ""

    size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    expected_rr: float = 0.0


class RiskManager:
    def __init__(
        self,
        risk_per_trade=0.01,      # 单笔最多亏 1%
        min_rr=1.5,               # 最小风险回报比
        min_stop_pct=0.01,        # 最小止损 1%
        max_stop_pct=0.03,        # 最大止损 3%
        min_take_pct=0.01,        # 最小止盈 1%
        atr_stop_mult=1.2,        # ATR 止损倍数
        atr_take_mult=2.0,        # ATR 止盈倍数
        lot_size=100,             # A 股 100 股
    ):
        self.risk_per_trade = risk_per_trade
        self.min_rr = min_rr
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.min_take_pct = min_take_pct
        self.atr_stop_mult = atr_stop_mult
        self.atr_take_mult = atr_take_mult
        self.lot_size = lot_size

    # ===================== 主入口 =====================
    def evaluate(
        self,
        last_price: float,
        chronos_low: float,
        chronos_high: float,
        atr: float,
        capital: float,
    ) -> TradePlan:
        if atr <= 0:
            return None
        # ---------- 止损候选 ----------
        sl_chronos = chronos_low
        sl_atr = last_price - self.atr_stop_mult * atr

        stop_loss = min(sl_chronos, sl_atr)

        # 止损上下限保护
        stop_loss = max(
            stop_loss,
            last_price * (1 - self.max_stop_pct),
        )
        stop_loss = min(
            stop_loss,
            last_price * (1 - self.min_stop_pct),
        )

        # ---------- 止盈候选 ----------
        tp_chronos = chronos_high
        tp_atr = last_price + self.atr_take_mult * atr
        tp_min = last_price * (1 + self.min_take_pct)

        take_profit = max(tp_chronos, tp_atr, tp_min)

        # ---------- 风险回报比 ----------
        risk = last_price - stop_loss
        reward = take_profit - last_price

        if risk <= 0 or reward <= 0:
            return TradePlan(
                allow_trade=False,
                reason="无效风险结构",
            )

        rr = reward / risk

        if rr < self.min_rr:
            return TradePlan(
                allow_trade=False,
                reason=f"RR 不足 ({rr:.2f})",
                expected_rr=rr,
            )

        # ---------- 仓位计算 ----------
        size = self._calc_size(
            capital=capital,
            risk_amount=capital * self.risk_per_trade,
            per_share_risk=risk,
        )

        if size <= 0:
            return TradePlan(
                allow_trade=False,
                reason="仓位为 0（风险过大）",
            )

        return TradePlan(
            allow_trade=True,
            size=size,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            expected_rr=round(rr, 2),
        )

    # ===================== 内部工具 =====================
    def _calc_size(self, capital, risk_amount, per_share_risk):
        raw_size = risk_amount / per_share_risk
        size = int(raw_size // self.lot_size) * self.lot_size
        return max(size, 0)
