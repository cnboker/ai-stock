from dataclasses import dataclass
from log import signal_log
from infra.core.config import settings


@dataclass
class TradePlan:
    allow_trade: bool
    reason: str = ""

    size: int = 0
    stop_loss: float = 0.0


# 允许什么（allow_open / allow_add / stop / take）
class RiskManager:
    def __init__(
        self
    ):
        self.lot_size = 100

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

            # 1. 计算止损位
            sl_atr = last_price - settings.ATR_STOP_MULT * (atr * last_price)
            stop_loss = min(chronos_low, sl_atr)

            # 2. 【复活参数】使用全局配置限制止损范围，不再硬编码 0.95/0.99
            limit_sl_far = last_price * (
                1 - settings.MAX_STOP_PCT
            )  # 比如最多容忍 8% 损耗
            limit_sl_near = last_price * (
                1 - settings.MIN_STOP_PCT
            )  # 至少要有 2% 保护空间

            stop_loss = max(stop_loss, limit_sl_far)
            stop_loss = min(stop_loss, limit_sl_near)

            # 3. 【新增盈利门槛】计算盈亏比 (RR)
            # 如果预期涨幅 (chronos_high) 连止损空间的 1.5 倍都不到，这单不接
            risk_dist = last_price - stop_loss
            reward_dist = chronos_high - last_price

            current_rr = reward_dist / risk_dist if risk_dist > 0 else 0

            if current_rr < settings.MIN_RR:
                return TradePlan(False, f"盈亏比太低: {current_rr:.2f}")

            # ---------- size ----------
            available_cash = capital  # 本次最大可用资金
            risk_per_share = last_price - stop_loss

            risk_cash = available_cash * settings.RISK_PER_TRADE

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
    
)
