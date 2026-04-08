from dataclasses import dataclass
from infra.core.runtime import GlobalState
from log import signal_log
from infra.core.dynamic_settings import settings


@dataclass
class TradePlan:
    allow_trade: bool
    reason: str = ""
    # 手数
    size: int = 0
    stop_loss: float = 0.0


# 允许什么（allow_open / allow_add / stop / take）
class RiskManager:
    def __init__(self):
        self.lot_size = 100

    # 交易可行性评估器
    # AI信号 → RiskManager.evaluate → TradePlan → PositionManager执行
    def evaluate(
        self,
        ticker: str,
        chronos_low: float,
        chronos_high: float,
        atr: float,
        capital: float,
        position_mgr,
    ) -> TradePlan:

        try:
            last_price = GlobalState.tickers_price[ticker]
            if ticker in position_mgr.cooldown:
                return TradePlan(allow_trade=False, reason="COOLDOWN")
            if atr <= 0:
                return TradePlan(False, "ATR 无效")

            # 1. 计算止损位
            sl_atr = last_price - settings.ATR_STOP * (atr * last_price)
            stop_loss = min(chronos_low, sl_atr)

            # 2. 【复活参数】使用全局配置限制止损范围，不再硬编码 0.95/0.99
            limit_sl_far = last_price * (1 - settings.MAX_STOP)  # 比如最多容忍 8% 损耗
            limit_sl_near = last_price * (1 - settings.MIN_STOP)  # 至少要有 2% 保护空间

            stop_loss = max(stop_loss, limit_sl_far)
            stop_loss = min(stop_loss, limit_sl_near)

            # 3. 【新增盈利门槛】计算盈亏比 (RR)
            # 如果预期涨幅 (chronos_high) 连止损空间的 1.5 倍都不到，这单不接
            risk_dist = last_price - stop_loss
            reward_dist = chronos_high - last_price

            current_rr = reward_dist / risk_dist if risk_dist > 0 else 0

            # ---------- size 计算逻辑修正 ----------
            available_cash = capital  # 比如 3000 元

            # 单股亏损空间 (例如 0.3 元)
            risk_per_share = last_price - stop_loss

            # 【关键修改】：风险金应基于账户总权益 (equity) 或 允许预算满仓
            # 如果你希望这 3000 元能全部花出去，这里应该增加风险容忍度
            # 或者直接使用账户总资产来计算 risk_cash
            total_equity = position_mgr.equity
            risk_cash = total_equity * settings.RISK
            # print(
            #     f"RiskManager: risk_cash={risk_cash}, risk_per_share={risk_per_share}, available_cash={available_cash} settings.RISK={settings.RISK}"
            # )
            # 1. 基于单笔最大亏损限制的股数
            risk_shares = int(risk_cash / risk_per_share) if risk_per_share > 0 else 0

            # 2. 基于本次预算金额限制的股数
            max_shares = int(available_cash / last_price)

            # 3. 取两者交集
            allowed_shares = min(risk_shares, max_shares)

            # 4. 修正 A 股手数逻辑：这里的 size 应该是股数，不是手数！
            # 如果你的下游执行函数需要的是“股数”，这里必须乘回 100
            theoretical_lots = allowed_shares // self.lot_size

            # 【新增：保底判断】
            if theoretical_lots <= 0:
                cost_one_lot = last_price * 100
                # 如果钱够买 100 股，且风险只超出了一点点（风险金的 1.5 倍以内）
                if available_cash >= cost_one_lot and (risk_per_share * 100) <= (risk_cash * 1.5):
                    theoretical_lots = 1  # 强行给 1 手，不让它因为微小的风控溢出而报错

            #final_shares = theoretical_lots * 100 # 换算回股数

            # if actual_lots <= 0:
            #     return TradePlan(False, "持仓过大，风控导致资金不足以购买一手(100股)")

            return TradePlan(
                True,
                size=theoretical_lots,  # <--- 注意：这里传回 (手数)
                stop_loss=round(stop_loss, 2),
            )
        except Exception:
            import traceback

            traceback.print_exc()
            return TradePlan(False, "风险计算异常")
            # ---------- 止损候选 ----------


risk_mgr = RiskManager()
