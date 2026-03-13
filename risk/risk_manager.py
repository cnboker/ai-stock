from dataclasses import dataclass
from typing import Optional

from log import signal_log


@dataclass
class TradePlan:
    allow_trade: bool
    reason: str = ""

    size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    expected_rr: float = 0.0

#允许什么（allow_open / allow_add / stop / take）
class RiskManager:
    def __init__(
        self,
        risk_per_trade=0.01,  # 单笔最多亏 1%
        min_rr=1.5,  # 最小风险回报比
        min_stop_pct=0.02,  # 最小止损 1%
        max_stop_pct=0.08,  # 最大止损 3%
        min_take_pct=0.01,  # 最小止盈 1%
        atr_stop_mult=1.2,  # ATR 止损倍数
        atr_take_mult=2.0,  # ATR 止盈倍数
        lot_size=100,  # A 股 100 股
    ):
        self.risk_per_trade = risk_per_trade
        self.min_rr = min_rr
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.min_take_pct = min_take_pct
        self.atr_stop_mult = atr_stop_mult
        self.atr_take_mult = atr_take_mult
        self.lot_size = lot_size

    # 交易可行性评估器
    # AI信号 → RiskManager.evaluate → TradePlan → PositionManager执行
    def evaluate(
        self,
        last_price: float,
        chronos_low: float,
        chronos_high: float,
        atr: float,
        capital: float,
    ) -> TradePlan:

        try:

            if atr <= 0:
                return TradePlan(False, "ATR 无效")

          # ---------- ATR ----------
            atr_price = atr

            # ---------- stop candidates ----------
            sl_chronos = chronos_low if chronos_low < last_price else last_price
            sl_atr = last_price - self.atr_stop_mult * atr_price

            # 用更紧的止损
            stop_loss = max(sl_chronos, sl_atr)

            # ---------- clamp stop ----------
            min_stop = last_price * (1 - self.max_stop_pct)
            max_stop = last_price * (1 - self.min_stop_pct)

            stop_loss = max(min(stop_loss, max_stop), min_stop)

            # ---------- take ----------
            tp_chronos = chronos_high if chronos_high > last_price else last_price
            tp_atr = last_price + self.atr_take_mult * atr_price
            tp_min = last_price * (1 + self.min_take_pct)

            take_profit = max(tp_chronos, tp_atr, tp_min)

            # ---------- max tp protection ----------
            max_take_pct = 0.25
            take_profit = min(take_profit, last_price * (1 + max_take_pct))

            # ---------- RR ----------
            risk = last_price - stop_loss
            reward = take_profit - last_price

            if risk <= 0 or reward <= 0:
                return TradePlan(False, "无效风险结构")

            rr = reward / risk

            # RR 上限保护
            max_rr = 3
            if rr > max_rr:
                take_profit = last_price + risk * max_rr
                reward = take_profit - last_price
                rr = reward / risk

            # RR 过滤
            if rr < self.min_rr:
                return TradePlan(False, f"RR 不足 ({rr:.2f})", expected_rr=rr)
            # ---------- size ----------
            available_cash = capital  # 本次最大可用资金
            risk_per_share = last_price - stop_loss

            max_shares = int(available_cash / last_price)
            risk_shares = int((available_cash * self.risk_per_trade) / risk_per_share)

            allowed_shares = min(max_shares, risk_shares)

            actual_shares = allowed_shares // self.lot_size          
            signal_log(f"price={last_price}, max_shares={max_shares},risk_shares={risk_shares}, size={actual_shares},actual_shares={actual_shares},stop_loss={stop_loss},take_profit={take_profit}, expected_rr={round(rr, 2)}")

            if actual_shares <= 0:
                return TradePlan(False, "仓位为 0")
            return TradePlan(
                True,
                size=actual_shares,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                expected_rr=round(rr, 2),
            )

        except Exception:
            import traceback
            traceback.print_exc()
            return TradePlan(False, "风险计算异常")
            # ---------- 止损候选 ----------



risk_mgr = RiskManager(
    risk_per_trade=0.02,  # 单笔最多亏 1%
    min_rr=1.5,  # 最低风险回报比
    min_stop_pct=0.01,  # 最小止损 1%
    max_stop_pct=0.03,  # 最大止损 3%
    min_take_pct=0.01,  # 最小止盈 1%
    atr_stop_mult=1.2,  # ATR 止损
    atr_take_mult=2.0,  # ATR 止盈
    lot_size=100,  # A 股
)
