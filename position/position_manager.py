from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from infra.core.runtime import RunMode
from infra.notify.order import OrderEvent, notify_order
from position.position import Position
from log import order_log, risk_log


class PositionManager:
    """
    PositionManager 只负责三件事：
    1. 账户资金
    2. 仓位状态
    3. 执行明确动作（OPEN / ADD / REDUCE / CLOSE / SHORT）
    """

    def __init__(self, init_cash: float = 0.0, run_mode: RunMode = RunMode.LIVE):
        # ===== 资金 =====
        self.cash: float = init_cash
        self.frozen_cash: float = 0.0
        self.run_mode: RunMode = run_mode
        self.account_name: str = ""

        # ===== 仓位 =====
        self.positions: Dict[str, Position] = {}

        # ===== 行情 =====
        self.price_cache: Dict[str, float] = {}

        # ===== 记录 =====
        self.trade_log: List[dict] = []
        self.last_action_ts: Dict[str, float] = {}        
    # ==================================================
    # 基础状态
    # ==================================================
    @property
    def equity(self) -> float:
        return self.cash + self.position_value()

    @property
    def available_cash(self) -> float:
        return self.cash

    def get(self, ticker: str) -> Optional[Position]:
        return self.positions.get(ticker)

    def pos_to_dict(self, ticker: str) -> dict:
        pos = self.get(ticker=ticker)
        if pos is None:
            return {}
        return {
            "position_size": getattr(pos, "size", 0),
            "direction": getattr(pos, "direction", ""),
            "entry_price": getattr(pos, "entry_price", 0),
            "stop_loss": getattr(pos, "stop_loss", 0),
            "take_profit": getattr(pos, "take_profit", 0),
        }

    def has_position(self, ticker: str) -> bool:
        pos = self.positions.get(ticker)
        return pos is not None and pos.size > 0

    def has_any_position(self) -> bool:
        for pos in self.positions.values():
            if pos.size != 0:
                return True
        return False

    def all_positions(self) -> Dict[str, Position]:
        return self.positions

    def position_value(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            price = self.price_cache.get(sym, pos.entry_price)
            total += pos.size * price * pos.contract_size
        return total
    
     # 检查止损止盈
    def check_stop_take(self, ticker: str, price: float) -> str | None:
        pos = self.positions.get(ticker)
        if not pos:
            return None
        if pos.size == 0:
            return None
        # LONG 仓位
        if pos.direction == "LONG":
            if pos.stop_loss is not None and price <= pos.stop_loss:
                return "STOP_LOSS"
            if pos.take_profit is not None and price >= pos.take_profit:
                return "TAKE_PROFIT"

        # SHORT 仓位
        elif pos.direction == "SHORT":
            if pos.stop_loss is not None and price >= pos.stop_loss:
                return "STOP_LOSS"
            if pos.take_profit is not None and price <= pos.take_profit:
                return "TAKE_PROFIT"

        return None
   
    # ==================================================
    # 行情
    # ==================================================
    def update_price(self, symbol: str, price: float):
        self.price_cache[symbol] = price

    def _require_price(self, symbol: str) -> float:
        price = self.price_cache.get(symbol)
        if price is None:
            raise RuntimeError(f"No price for {symbol}")
        return price

    # ==================================================
    # 交易入口（供 execute_equity_action 调用）
    # ==================================================
    def open_or_add(
        self,
        *,
        ticker: str,
        strength: float = 0.0,
        price: float,
        reason: str,
        contract_size: int = 100,
        size: int = None,
        stop_loss: float = None,
        take_profit: float = None,
        ):
        """
        打开新仓或加仓
        支持两种方式：
        1️⃣ 直接传 size → 精确仓位
        2️⃣ 传 strength → 按 equity * strength 自动计算仓位
        stop_loss / take_profit 可直接指定
        """

        pos = self.positions.get(ticker)
        if pos:
            # ==== 加仓 ====
            add_size = 0
            if size is not None:
                add_size = size
            elif strength > 0:
                add_size = self._calc_add_size(pos, strength, price)

            if add_size <= 0:
                return

            cost = add_size * price * pos.contract_size
            if cost > self.cash:
                order_log(f"{ticker} ADD rejected (cash insufficient)")
                return

            # 加权平均成本
            pos.entry_price = (pos.entry_price * pos.size + price * add_size) / (pos.size + add_size)
            pos.size += add_size
            self.cash -= cost

            # 更新止损止盈
            if stop_loss is not None:
                pos.stop_loss = stop_loss
            if take_profit is not None:
                pos.take_profit = take_profit

            self._record(
                symbol=ticker,
                action="ADD",
                size=add_size,
                price=price,
                value=cost,
                reason=reason,
            )

        else:
            # ==== 开新仓 ====
            open_size = 0
            if size is not None:
                open_size = size
            elif strength > 0:
                open_size = self._calc_open_size(strength, price, contract_size)

            if open_size <= 0:
                return

            cost = open_size * price * contract_size
            if cost > self.cash:
                order_log(f"{ticker} OPEN rejected (cash insufficient)")
                return

            pos = Position(
                ticker=ticker,
                direction="LONG",
                size=open_size,
                entry_price=price,
                open_time=datetime.now(),
                contract_size=contract_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self.positions[ticker] = pos
            self.cash -= cost

            self._record(
                symbol=ticker,
                action="OPEN",
                size=open_size,
                price=price,
                value=cost,
                reason=reason,
            )

    def open_short(
        self,
        *,
        ticker: str,
        strength: float,
        price: float,
        reason: str,
    ):
        # 目前简化为 CLOSE（如后续支持真做空再扩展）
        self.close(ticker=ticker, price=price, reason=reason)

    def reduce(
        self,
        *,
        ticker: str,
        strength: float,
        price: float,
        reason: str
    ):
        self._reduce(ticker, strength, price, reason)

    def close(
        self,
        *,
        ticker: str,
        price: float,
        reason: str,
    ):
        self._close(ticker, price, reason)

    # ==================================================
    # 内部执行逻辑
    # ==================================================
    def _open(
        self,
        ticker: str,
        strength: float,
        price: float,
        reason: str,
        contract_size: int=100,
    ):
        size = self._calc_open_size(strength, price, contract_size)
        if size <= 0:
            return

        cost = size * price * contract_size
        if cost > self.cash:
            order_log(f"{ticker} OPEN rejected (cash insufficient)")
            return

        self.cash -= cost

        pos = Position(
            ticker=ticker,
            direction="LONG",
            size=size,
            entry_price=price,
            open_time=datetime.now(),
            contract_size=contract_size,
        )
        self.positions[ticker] = pos

        self._record(
            symbol=ticker,
            action="OPEN",
            size=size,
            price=price,
            value=cost,
            reason=reason,
        )

    def _add(
        self,
        ticker: str,
        strength: float,
        price: float,
        reason: str,
    ):
        pos = self.positions.get(ticker)
        if not pos:
            return

        size = self._calc_add_size(pos, strength, price)
        if size <= 0:
            return

        cost = size * price * pos.contract_size
        if cost > self.cash:
            order_log(f"{ticker} ADD rejected (cash insufficient)")
            return

        # 加权成本
        pos.entry_price = (pos.entry_price * pos.size + price * size) / (
            pos.size + size
        )
        pos.size += size
        self.cash -= cost

        self._record(
            symbol=ticker,
            action="ADD",
            size=size,
            price=price,
            value=cost,
            reason=reason,
        )

    def _reduce(
        self,        
        ticker: str,
        strength: float = None,
        price: float = None,
        reason: str = "",
        size: int = 0,  # 新增参数，用于精确减仓
    ):
        """
        减仓逻辑
        size 优先级最高：直接减指定数量
        strength 次之：按比例减仓
        """
        pos = self.positions.get(ticker)
        if not pos:
            return

        # ===== 1️⃣ 优先使用精确 size =====
        if size is not None and size > 0:
            reduce_size = min(size, pos.size)
        elif strength is not None:
            # strength ∈ [0, 1] 表示减仓比例
            ratio = min(max(strength, 0.0), 1.0)
            reduce_size = int(pos.size * ratio)
        else:
            # 默认减 30%
            reduce_size = int(pos.size * 0.3)

        if reduce_size <= 0:
            return

        # ===== 2️⃣ 执行减仓 =====
        pos.size -= reduce_size
        cash_back = reduce_size * price * pos.contract_size
        self.cash += cash_back

        self._record(
            symbol=ticker,
            action="REDUCE",
            size=reduce_size,
            price=price,
            value=cash_back,
            reason=reason,
        )

        # ===== 3️⃣ 仓位清空时移除 =====
        if pos.size <= 0:
            del self.positions[ticker]

    def _close(
            self,
            ticker: str,
            price: float,
            reason: str,
        ):
            pos = self.positions.get(ticker)
            if not pos:
                return
            if pos.size == 0:
                return
            value = pos.size * price * pos.contract_size
            self.cash += value
            pos.size = 0
            self._record(
                symbol=ticker,
                action="CLOSE",
                size=pos.size,
                price=price,
                value=value,
                reason=reason,
            )

    # ==================================================
    # Size 计算（模拟 / 实盘统一入口）
    # ==================================================
    def _calc_open_size(
        self,
        strength: float,
        price: float,
        contract_size: int=100,
    ) -> int:
        """
        strength ∈ (0, 1] 表示占用 equity 比例
        """
        budget = self.equity * min(max(strength, 0.0), 1.0)
        size = int(budget / (price * contract_size))
        return max(size, 0)

    def _calc_add_size(
        self,
        pos: Position,
        strength: float,
        price: float,
    ) -> int:
        budget = self.equity * min(max(strength, 0.0), 1.0)
        size = int(budget / (price * pos.contract_size))
        return max(size, 0)

    # ==================================================
    # 记录 & 通知
    # ==================================================
    def _record(
        self,
        *,
        symbol: str,
        action: str,
        size: int,
        price: float,
        value: float,
        reason: str,
    ):
        ts = datetime.now()

        self.trade_log.append(
            {
                "ts": ts,
                "symbol": symbol,
                "action": action,
                "size": size,
                "price": price,
                "value": value,
                "reason": reason,
                "run_mode": self.run_mode,
            }
        )

        notify_order(
            OrderEvent(
                ts=ts,
                symbol=symbol,
                action=action,
                side="LONG",
                size=size,
                price=price,
                value=value,
                reason=reason,
            ),
            self,
        )

    def load_from_yaml(self, data):
        self.positions.clear()
        self.cash = data.get("cash", 100000)
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

    def clear(self):
        self.positions.clear()
        self.trade_log.clear()
