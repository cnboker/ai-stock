from datetime import datetime
from typing import Dict, Optional

import yaml
from .position import Position

'''
1 维护当前持仓（你只能手动给，它就当真）
2 新信号来时：开 / 加 / 忽略 / 平
3 监控止损 / 止盈
4 输出“交易指令”（order），不直接下单
'''

class PositionManager:
    def __init__(self):
        # ticker -> Position
        self.positions: Dict[str, Position] = {}

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
        return self._manage_existing_position(
            pos, signal, last_price, trade_plan
        )

    # ===================== 开仓 =====================
    def _open_long(self, ticker, plan, price):
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

            for ticker, p in data.get("positions", {}).items():
                self.positions[ticker] = Position(
                    ticker=ticker,
                    direction=p["direction"],
                    size=p["size"],
                    entry_price=p["entry_price"],
                    stop_loss=p["stop_loss"],
                    take_profit=p["take_profit"],
                    
                )
            print('positions', self.positions)
            return data.get("account", {})