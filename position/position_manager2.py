from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from position.PositionPolicy import PositionAction
from position.position import Position
from log import order_log, risk_log

'''
PositionManager：只做三件事

持仓状态

资金状态

执行 Action
'''
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

    def _open(self, symbol: str, action: PositionAction):
        price = self.price_cache.get(symbol)
        if price is None:
            raise RuntimeError(f"No price for {symbol}")

        size = action.size
        cost = size * price * action.contract_size

        if cost > self.cash:
            order_log(f"{symbol} OPEN rejected (insufficient cash)")
            return

        self.cash -= cost

        self.positions[symbol] = Position(
            ticker=symbol,
            direction=action.side,
            size=size,
            entry_price=price,
            stop_loss=action.stop_loss,
            take_profit=action.take_profit,
            open_time=datetime.now(),
            contract_size=action.contract_size,
        )

        order_log(f"{symbol} OPEN {action.side} size={size} price={price}")

  # ======================================================
    # 主交易入口（唯一产生 OPEN / ADD / REVERSE 的地方）
    # ======================================================
    def on_signal(
        self,
        symbol: str,
        signal: str,           # LONG / SHORT / HOLD
        last_price: float,
        trade_plan=None,       # 来自 RiskManager
    ) -> Optional[PositionAction]:

        pos = self.positions.get(symbol)

        # ================= 无仓位 =================
        if pos is None or pos.size <= 0:
            if signal == "LONG" and trade_plan:
                return PositionAction(
                    action="OPEN",
                    ratio=1.0,          # 对 OPEN 可忽略
                )
            return None

        # ================= 有仓位 =================
        # ---- 反向信号：清仓（是否反手，由你以后决定）----
        if signal == "SHORT":
            return PositionAction("CLOSE")

        # ---- 可选：加仓逻辑 ----
        if signal == "LONG" and trade_plan and trade_plan.allow_add:
            return PositionAction(
                action="OPEN",          # 复用 OPEN 表示加仓
                ratio=trade_plan.add_ratio,
            )

        return None
    
    def _reduce(self, symbol: str, ratio: float):
        pos = self.positions[symbol]
        price = self.price_cache.get(symbol, pos.entry_price)

        reduce_size = pos.size * ratio
        if reduce_size <= 0:
            return

        pos.size -= reduce_size
        cash_back = reduce_size * price * pos.contract_size
        self.cash += cash_back

        order_log(
            f"{symbol} REDUCE {ratio:.2f} size={reduce_size} price={price}"
        )

        if pos.size <= 0:
            del self.positions[symbol]


    def _close(self, symbol: str):
        pos = self.positions.pop(symbol)
        price = self.price_cache.get(symbol, pos.entry_price)

        value = pos.size * price * pos.contract_size
        self.cash += value

        order_log(
            f"{symbol} CLOSE size={pos.size} price={price}"
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