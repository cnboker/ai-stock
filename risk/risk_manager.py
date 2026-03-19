from dataclasses import dataclass
from typing import Optional

from log import signal_log


@dataclass
class TradePlan:
    allow_trade: bool
    reason: str = ""

    size: int = 0
    stop_loss: float = 0.0


# 允许什么（allow_open / allow_add / stop / take）
class RiskManager:
    def __init__(
        self,
        risk_per_trade=0.02,  # 单笔最多亏 1%
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
        ticker: str,
        last_price: float,
        chronos_low: float,
        chronos_high: float,
        atr: float,
        capital: float,
        position_mgr,
    ) -> TradePlan:

        try:
            if ticker in position_mgr.cooldown:
                return TradePlan(allow_trade=False, reason="COOLDOWN")
            if atr <= 0:
                return TradePlan(False, "ATR 无效")
          
            # ✅ 1. ATR 不要乘 price           
            atr_price = atr * last_price
            # ✅ 2. 两种止损
            sl_chronos = chronos_low
            sl_atr = last_price - self.atr_stop_mult * atr_price

            # ✅ 3. 取“更保守”（更远）的那个
            stop_loss = min(sl_chronos, sl_atr)

            # ✅ 4. 只做“极端保护”（防bug）
            max_risk = last_price * 0.95   # 最多亏5%
            min_risk = last_price * 0.99  # 至少1%

            stop_loss = max(stop_loss, max_risk)   # 不允许太远
            stop_loss = min(stop_loss, min_risk)   # 不允许太近

            # ---------- size ----------
            available_cash = capital  # 本次最大可用资金
            risk_per_share = last_price - stop_loss

            risk_cash = available_cash * self.risk_per_trade

            risk_shares = int(risk_cash / risk_per_share)

            max_shares = int(available_cash / last_price)

            allowed_shares = min(risk_shares, max_shares)

            actual_shares = allowed_shares // self.lot_size
            signal_log(
                f"price={last_price}, max_shares={max_shares},risk_shares={risk_shares}, size={actual_shares},actual_shares={actual_shares},stop_loss={stop_loss}"
            )

            if actual_shares <= 0:
                return TradePlan(False, "仓位为 0")
            return TradePlan(
                True,
                size=actual_shares,
                stop_loss=round(stop_loss, 2),
            )

        except Exception:
            import traceback

            traceback.print_exc()
            return TradePlan(False, "风险计算异常")
            # ---------- 止损候选 ----------


risk_mgr = RiskManager(
    risk_per_trade=0.02,  # 单笔最多亏 1%
    min_rr=2,  # 最低风险回报比
    min_stop_pct=0.02,  # 最小止损 1%
    max_stop_pct=0.05,  # 最大止损 3%
    min_take_pct=0.01,  # 最小止盈 1%
    atr_stop_mult=1.5,  # ATR 止损
    atr_take_mult=4.0,  # ATR 止盈
    lot_size=100,  # A 股
)
