from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from infra.notify.order import OrderEvent, notify_order
from position.PositionPolicy import PositionAction
from position.position import Position
from log import order_log, risk_log

"""
PositionManager：只做三件事

持仓状态

资金状态

执行 Action
"""


# 算钱、改仓位、记账
class PositionManager:
    def __init__(self, init_cash: float = 0.0):
        # ===== 账户资金 =====
        self.cash: float = init_cash
        self.frozen_cash: float = 0.0
        self.account_name: str = ""

        # ===== 持仓 =====
        self.positions: Dict[str, Position] = {}

        # ===== 价格缓存（由行情系统更新）=====
        self.price_cache: Dict[str, float] = {}

        # ===== 统计 / 心跳 =====
        self.last_action_ts: Dict[str, float] = {}

    @property
    def equity(self) -> float:
        return self.cash + self.frozen_cash + self.position_value()

    @property
    def available_cash(self) -> float:
        return self.cash

    def update_price(self, symbol: str, price: float):
        print(type(price))
        self.price_cache[symbol] = price

    def position_value(self) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = self.price_cache.get(symbol, pos.entry_price)
            total += pos.size * price * pos.contract_size
        return total

    def _log_action(self, symbol, action, extra: str | None = None):
        """
        统一动作日志（Policy / Signal / Gate）
        """
        msg = f"[PositionAction] {symbol} → {action.action}"

        if hasattr(action, "ratio") and action.ratio > 0:
            msg += f" ratio={action.ratio:.2f}"

        if extra:
            msg += f" | {extra}"

        order_log(msg)

    # ======================================================
    # 主交易入口（唯一产生 OPEN / ADD / REVERSE 的地方）
    # ======================================================
    def on_signal(self, symbol: str, action: str, confidence: float, last_price: float, trade_plan=None):
        pos = self.positions.get(symbol)
        if not pos or pos.size <= 0:
            if action == "LONG" and trade_plan and trade_plan.allow_open:
                return PositionAction(action="OPEN", plan=trade_plan)
            return None

        # ---------------- REDUCE 分批 ----------------
        if action == "REDUCE":
            reduce_ratio = min(1.0, confidence)  # confidence 控制减仓比例
            reduce_size = int(pos.size * reduce_ratio // 100 * 100)  # 向下取整到 100 股
            if reduce_size > 0:
                return PositionAction(action="REDUCE", size=reduce_size)

        # ---------------- SHORT / CLOSE ----------------
        if action == "SHORT":
            return PositionAction(action="CLOSE")

        # ---------------- ADD ----------------
        if action == "LONG" and trade_plan and trade_plan.allow_add:
            add_budget = self.get_add_budget(symbol, trade_plan)
            size = int(add_budget / last_price // 100 * 100)
            if size > 0:
                return PositionAction(action="ADD", size=size, plan=trade_plan)

        return None

    def apply_action(self, symbol: str, action: PositionAction):
        self._log_action(symbol, action)
        self.last_action_ts[symbol] = datetime.now().timestamp()

        pos = self.positions.get(symbol)

        match action.action:
            case "HOLD":
                self._on_hold(symbol, action)
            case "OPEN":
                if pos and pos.size > 0:
                    order_log(f"{symbol} OPEN ignored (position exists)")
                    return
                self._open(symbol, action)
            case "ADD":
                self._add(symbol, action)
            case "REDUCE":
                if not pos or pos.size <= 0:
                    raise RuntimeError(f"{symbol} REDUCE but no position")
                self._reduce(symbol, action.reduce_ratio)
            case "CLOSE":
                if not pos or pos.size <= 0:
                    raise RuntimeError(f"{symbol} CLOSE but no position")
                self._close(symbol)

            case "REVERSE":
                if not pos or pos.size <= 0:
                    raise RuntimeError(f"{symbol} REVERSE but no position")
                self._close(symbol)
                self._open(symbol, action)

            case _:
                raise ValueError(f"Unknown action {action.action}")

    def _on_hold(self, symbol: str, action: PositionAction):
        risk_log(f"{symbol} HOLD")

    # 开仓
    def _open(self, symbol: str, action: PositionAction):
        price = self.price_cache.get(symbol)
        price = self._require_price(symbol)
        plan = self._require_plan(action)

        size = action.size
        cost = size * price * action.contract_size

        if cost > self.cash:
            order_log(f"{symbol} OPEN rejected (insufficient cash)")
            return

        self.cash -= cost

        self.positions[symbol] = Position(
            ticker=symbol,
            direction="LONG",
            size=action.size,
            entry_price=price,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            open_time=datetime.now(),
            contract_size=action.contract_size,
        )
        value = size * price * action.contract_size
        # order_log(f"{symbol} OPEN {action.side} size={size} price={price}")
        notify_order(
            OrderEvent(
                ts=datetime.now(),
                symbol=symbol,
                action="OPEN",
                side="LONG",
                size=size,
                price=price,
                value=value,
                reason="Signal LONG + allow_open",
                extra={
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                },
            ),
            self,
        )

    # =========================
    # ADD：加仓
    # =========================
    def _add(self, symbol: str, action: PositionAction):
        pos = self.positions.get(symbol)
        if not pos:
            raise RuntimeError(f"{symbol} ADD but no position")

        price = self._require_price(symbol)

        size = action.size
        cost = size * price * pos.contract_size
        if cost > self.cash:
            order_log(f"{symbol} ADD rejected (cash insufficient)")
            return

        # 加权成本
        total_size = pos.size + size
        pos.entry_price = (pos.entry_price * pos.size + price * size) / total_size

        pos.size = total_size
        self.cash -= cost
        value = size * price * action.contract_size
        # order_log(f"{symbol} ADD size={size} price={price}")
        notify_order(
            OrderEvent(
                ts=datetime.now(),
                symbol=symbol,
                action="ADD",
                side="LONG",
                size=size,
                price=price,
                value=value,
                reason="Pyramid add",
            ),
            self,
        )

    def _reduce(self, symbol: str, ratio: float):
        pos = self.positions[symbol]
        if not pos:
            return
        price = self._require_price(symbol)

        reduce_size = pos.size * ratio
        if reduce_size <= 0:
            return

        pos.size -= reduce_size
        cash_back = reduce_size * price * pos.contract_size
        self.cash += cash_back

        # order_log(
        #     f"{symbol} REDUCE {ratio:.2f} size={reduce_size} price={price}"
        # )
        value = reduce_size * price * pos.contract_size
        notify_order(
            OrderEvent(
                ts=datetime.now(),
                symbol=symbol,
                action="REDUCE",
                side="LONG",
                size=pos.size,
                price=price,
                value=value,
                reason="REDUCE hit",
            ),
            self,
        )

        if pos.size <= 0:
            del self.positions[symbol]

    def _close(self, symbol: str):
        pos = self.positions.pop(symbol)
        price = self.price_cache.get(symbol, pos.entry_price)

        value = pos.size * price * pos.contract_size
        self.cash += value

        # order_log(
        #     f"{symbol} CLOSE size={pos.size} price={price}"
        # )

        notify_order(
            OrderEvent(
                ts=datetime.now(),
                symbol=symbol,
                action="CLOSE",
                side="LONG",
                size=pos.size,
                price=price,
                value=value,
                reason="StopLoss hit",
            ),
            self,
        )

    def _require_price(self, symbol: str) -> float:
        price = self.price_cache.get(symbol)
        if price is None:
            raise RuntimeError(f"No price for {symbol}")
        return price

    def _require_plan(self, action: PositionAction):
        if not action.plan:
            raise RuntimeError("OPEN/ADD requires trade_plan")
        return action.plan

    def get_add_budget(
        self,
        ticker: str,
        trade_plan,
    ) -> float:
        position = self.positions.get(ticker)
        if not position:
            return 0.0

        # 当前仓位市值
        current_value = position.market_value

        # 单票最大允许市值
        max_value = self.equity * trade_plan.single_position_limit

        # 还能加多少
        remaining_capacity = max(max_value - current_value, 0)

        # 策略希望加多少（比例）
        desired_add = current_value * trade_plan.max_signal_pct

        # 最终取最小
        return min(
            remaining_capacity,
            desired_add,
            self.available_cash,
        )

    def load_from_yaml(self, data):
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
