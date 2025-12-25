from datetime import datetime
from typing import Dict, Optional

import yaml

from data.loader import fetch_sina_quote_live
from .position import Position

"""
1 维护当前持仓（你只能手动给，它就当真）
2 新信号来时：开 / 加 / 忽略 / 平
3 监控止损 / 止盈
4 输出“交易指令”（order），不直接下单

账户名	account_name
当前现金	cash
可下单资金	available_cash
持仓市值	position_value
账户总权益	equity
本次信号预算	signal_capital

"""


class PositionManager:

    def __init__(self, init_cash: float = 0.0):
        # ticker -> Position
        self.positions: Dict[str, Position] = {}

        # ===== 资金相关 =====
        self.cash: float = init_cash  # 可用现金
        self.frozen_cash: float = 0.0  # 冻结资金（预留）
        self.account_name: str = ""

    # 冻结资金
    """
    字段	含义
    cash	账户可用现金（未冻结）
    frozen_cash	已下单但未成交的冻结资金
    positions	当前持仓
    position_value	持仓市值（按最新价）
    equity	账户总权益
    #equity = cash + frozen_cash + position_value
    #available_cash = cash
    """


    @property
    def equity(self) -> float:
        return self.cash + self.frozen_cash + self.position_value()

    @property
    def available_cash(self) -> float:
        return self.cash

     # ===================== 单个仓位市值 =====================
    def market_value(self, ticker:str, latest_price: float) -> float:
        pos = self.positions[ticker]
        return pos.size * latest_price * 100

    # ===================== 所有持仓市值 =====================
    def position_value(self) -> float:
        #price_map: dict[str, float] | None = None
        price_map = {}            
        # 1. 获取所有 ticker
        tickers = [pos.ticker for pos in self.positions.values()]

        # 3. 调用外部方法获取最新价格
        latest_prices = fetch_sina_quote_live(tickers)  # 返回 dict: {ticker: price}
        print("latest_prices", latest_prices)
        # 4. 更新 price_map
        price_map.update(latest_prices)

        # 5. 计算总持仓市值
        return sum(
            self.market_value(pos.ticker, price_map.get(pos.ticker, pos.entry_price))
            for pos in self.positions.values()
        )

    # ========= 仓位管理 =========
    def add_position(self, pos: Position):
        cost = pos.size * pos.entry_price
        if cost > self.cash:
            raise ValueError("Insufficient cash")

        self.cash -= cost
        self.positions[pos.ticker] = pos

    def close_position(self, ticker: str, price: float):
        pos = self.positions.pop(ticker, None)
        if not pos:
            return

        value = pos.size * price
        self.cash += value

    # ===================== 手动注入仓位 =====================
    def load_manual_positions(self, positions: Dict[str, dict]):
        """
        positions = {
            "600519": {
                "size": 100, #手
                "entry_price": 1680, #开仓价格
                "stop_loss": 1650, #停损价格
                "take_profit": 1750, #止盈价
            }
        }
        """
        for ticker, p in positions.items():
            self.positions[ticker] = Position(
                ticker=ticker,
                direction="LONG",
                size=p["size"],
                entry_price=p["entry_price"],
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
                open_time=datetime.now(),
            )

    # ===================== 主决策入口 =====================
    def on_signal(
        self,
        ticker: str,
        signal: str,
        last_price: float,
        trade_plan=None,  # 来自 RiskManager
    ) -> Optional[dict]:

        pos = self.positions.get(ticker)

        # ---------- 无仓位 ----------
        if pos is None:
            if signal == "LONG" and trade_plan:
                return self._open_long(ticker, trade_plan, last_price)
            return None

        # ---------- 有仓位 ----------
        return self._manage_existing_position(pos, signal, last_price, trade_plan)

    # ===================== 开仓 让“开仓”真正消耗资金（核心）=====================
    def _open_long(self, ticker, plan, price):
        cost = plan.size * price

        if cost > self.cash:
            return None  # 或抛异常

        self.cash -= cost

        pos = Position(
            ticker=ticker,
            direction="LONG",
            size=plan.size,
            entry_price=price,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            open_time=datetime.now(),
        )

        self.positions[ticker] = pos

        return {
            "action": "OPEN",
            "ticker": ticker,
            "direction": "LONG",
            "size": plan.size,
            "price": price,
            "stop_loss": plan.stop_loss,
            "take_profit": plan.take_profit,
        }

    # ===================== 持仓管理 =====================
    def _manage_existing_position(
        self, pos: Position, signal: str, price: float, plan
    ) -> Optional[dict]:

        # ---- 止损 ----
        if price <= pos.stop_loss:
            return self._close_position(pos, price, "STOP_LOSS")

        # ---- 止盈 ----
        if price >= pos.take_profit:
            return self._close_position(pos, price, "TAKE_PROFIT")

        # ---- 反向信号 ----
        if signal == "SHORT":
            return self._close_position(pos, price, "REVERSE_SIGNAL")

        # ---- 加仓逻辑（可选）----
        if signal == "LONG" and plan:
            if plan.expected_rr > 2.5:
                return self._add_position(pos, plan, price)

        return None

    # ===================== 平仓 =====================
    def _close_position(self, pos: Position, price, reason):
        value = pos.size * price
        self.cash += value

        del self.positions[pos.ticker]

        return {
            "action": "CLOSE",
            "ticker": pos.ticker,
            "size": pos.size,
            "price": price,
            "reason": reason,
        }

    # ===================== 加仓 =====================
    def _add_position(self, pos: Position, plan, price):
        add_size = plan.size // 2
        pos.size += add_size
        pos.stop_loss = max(pos.stop_loss, plan.stop_loss)
        pos.take_profit = plan.take_profit

        return {
            "action": "ADD",
            "ticker": pos.ticker,
            "size": add_size,
            "price": price,
        }

    def load_from_yaml(self, path="data/live_positions.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.positions.clear()
        self.cash = data.get("init_cash", 100000)
        self.account_name = data.get("account", "")

        for ticker, p in data.get("positions", {}).items():
            cost = p["size"] * p["entry_price"]
            self.cash -= cost
            self.positions[ticker] = Position(
                ticker=ticker,
                direction=p["direction"],
                size=p["size"],
                entry_price=p["entry_price"],
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
            )


position_mgr = PositionManager()

# 1️ 启动时加载真实仓位
account = position_mgr.load_from_yaml("config/live_positions.yaml")
