from datetime import datetime
from typing import Dict, List, Optional
from global_state import state_lock
from infra.core.runtime import GlobalState, RunMode
from infra.notify.order import OrderEvent, notify_order
from infra.persistence.live_positions import persist_live_positions
from position.position import Position
from log import order_log, risk_log
from infra.core.dynamic_settings import settings

class PositionManager:
    """
    PositionManager 只负责三件事：
    1. 账户资金
    2. 仓位状态
    3. 执行明确动作（OPEN / ADD / REDUCE / CLOSE / SHORT）
    """

    def __init__(self, init_cash: float = 0.0):
        # ===== 资金 =====
        self.cash: float = init_cash
        self.frozen_cash: float = 0.0
        self.account_name: str = ""
        self.max_occupied = 0.0  # 回测时统计最大占用率
        # ===== 仓位 =====
        self.positions: Dict[str, Position] = {}

        # ===== 行情 =====
        self.price_cache: Dict[str, float] = {}

        # ===== 记录 =====
        self.trade_log: List[dict] = []
        self.last_action_ts: Dict[str, float] = {}        
        self.cooldown = {}  # ticker -> 剩余bar数
        self.total_trade_count = 0  # 新增：用于回测统计总成交次数
        self.watchlist = {}  # 观察池 
        self.current_market_time = None  # 当前处理的 K 线时间戳

    def update_cooldown(self):
        for k in list(self.cooldown.keys()):
            self.cooldown[k] -= 1
            if self.cooldown[k] <= 0:
                del self.cooldown[k]        
    # ==================================================
    # 基础状态
    # ==================================================
    @property
    def equity(self) -> float:
        return self.cash + self.position_value()

    @property
    def available_cash(self) -> float:
        return self.cash

    def get_tickers_from_positions_and_watchlist(self):
        with state_lock:
            active_positions = list(self.positions.keys()) # [ticker, ...]
            watch_tickers = list(self.watchlist.keys())
            tickers = list(set(active_positions + watch_tickers))
        return tickers

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
    
    def get_ticker_value(self, ticker, price):
        pos = self.positions.get(ticker)
        if not pos or pos.size <= 0:
            return 0.0
        return pos.size * price * pos.contract_size
    
    def position_value(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            price = self.price_cache.get(sym, pos.entry_price)
            total += pos.size * price * pos.contract_size
            #print(f"DEBUG: {sym} 持仓 {pos.size} 手, 现价 {price}, 乘数 {pos.contract_size}, 计算市值: {pos.size * price * pos.contract_size}")
        return total
    
    def to_dict(self) -> Dict:
        data = {
            "account": self.account_name,
            "cash": round(self.cash,2),
            "positions": {},
            }

        for symbol, pos in self.positions.items():
            data["positions"][symbol] = {
                "direction": pos.direction,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "stop_loss": pos.stop_loss,
                "highest_price": pos.highest_price,            
                "tp1_hit": pos.tp1_hit,
                "open_time": pos.open_time.isoformat(),
            }
        return data
    
    #移动止损核心函数
    def update_trailing_stop(self, ticker, price, atr):
        position = self.positions.get(ticker)
        #print(f"position={position}")
        # 增加对 price 的有效性检查
        if not position or atr is None or price is None:
            return
        if position.highest_price is None :
            print(f"\n[DEBUG] 捕获到 NoneType 污染!")
            print(f"Ticker: {ticker}")
            print(f"Position Object: {position.__dict__}") # 打印对象所有属性
            print(f"Input Price: {price}, ATR: {atr}")
            # 主动抛出异常，带上更多信息
            raise ValueError(f"CRITICAL: {ticker} has None in its fields!")
        entry = position.entry_price
        
        # 1. 稳健初始化最高价 (修复报错点)
        current_highest = position.highest_price if position.highest_price is not None else 0.0
        if price > current_highest:
            position.highest_price = float(price)

        # 2. 计算止损基础
        # 确保 stop 有初始值，防止后面 max(stop, ...) 报错
        stop = position.stop_loss if position.stop_loss is not None else (entry * 0.95)
        
        # ATR 逻辑保持不变
        atr_price = atr * price
        profit = (price - entry) / entry if entry > 0 else 0

        # =============================
        # 状态切换
        # =============================
        if position.stage == "init" and profit > settings.INIT_PT:
            position.stage = "profit_lock"

        if position.stage == "profit_lock" and profit > settings.TREND_STAGE:
            position.stage = "trend"

        # =============================
        # 各阶段逻辑 (逻辑重构：确保 stop 始终不下降)
        # =============================

        # 🟢 阶段1：刚盈利 → 移动到开仓价保本
        if position.stage == "profit_lock":
            # 既然已经赚了 3% 了，与其回落到 0% 止损，
            # 不如锁住一半利润，比如保住 1.5% 的利润
            lock_profit_price = entry + (price - entry) * 0.5 
            stop = max(stop, lock_profit_price)

        # 🟡 阶段2：大肉趋势 → 移动 ATR 跟踪止损
        elif position.stage == "trend":
            # 1. 确保最高价有效，如果为 None 则降级使用现价
            h_price = position.highest_price if position.highest_price is not None else price
            
            # 2. 计算基于 ATR 的跟踪止损线
            trail_stop = h_price - (settings.ATR_MULT * atr_price)
            
            # 3. 确保 stop 有数值（如果之前 stop_loss 是 None，则取一个保底值）
            current_stop = stop if (stop is not None and stop > 0) else (entry * 0.95)
            
            # 4. 只允许止损线上移，绝不下移
            stop = max(current_stop, trail_stop)

        # 3. 最终赋值 (A股价格建议保留 3 位精度给 ETF，或者 2 位给股票)
        # 注意：sz159908 是 ETF，建议用 round(stop, 3)
        precision = 3 if ticker.startswith(('sz15', 'sh51')) else 2
        position.stop_loss = round(float(stop), precision)
        
    # 移动止盈（分批止盈）
    def check_take_profit(self, ticker, price):
        position = self.positions.get(ticker)
        if not position:
            return None

        entry = position.entry_price

        # 👉 提高止盈阈值（关键）
        tp1 = entry * settings.TP1
        tp2 = entry * settings.TP2

        # 🟢 只在趋势阶段才允许止盈
        if position.stage != "trend":
            return None

        # TP1：只减30%
        if not position.tp1_hit and price >= tp1:
            position.tp1_hit = True
            return 0.5

        # TP2：再减30%
        if not position.tp2_hit and price >= tp2:
            position.tp2_hit = True
            return 0.3

        return None

    # ==================================================
    # 行情
    # ==================================================
    def update_price(self, symbol: str, price: float):
        self.price_cache[symbol] = price

    def get_price(self, symbol: str) -> float:
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
        prediction_id: Optional[int] = None,      
        ):
        """
        打开新仓或加仓
        支持两种方式：
        1️⃣ 直接传 size → 精确仓位
        2️⃣ 传 strength → 按 equity * strength 自动计算仓位
        stop_loss  可直接指定
        """

        pos = self.positions.get(ticker)
        if pos and pos.size > 0:
          
            # ==== 加仓 ====
            if pos.tp1_hit:
                print(f"⚠️ {ticker} 已触及 TP1，当前价格 {price}，加仓操作将被拒绝以保护利润。")
                return
            # 🚨 价格步长检查：当前价必须比上次成交价高出 1.5%
            if price < pos.entry_price * 1.015 :
                # print(f"跳过加仓：当前价 {price} 未达到持仓成本高出 1.5% {pos.entry_price * 1.015} 的要求")
                return 
        
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

            self._record(
                symbol=ticker,
                action="ADD",
                size=add_size,
                price=price,
                value=cost,
                reason=reason,
                prediction_id=prediction_id
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
                highest_price=price,          
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
                prediction_id=prediction_id
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
        reason: str,
        prediction_id: Optional[int]=None,
    ):
        self._reduce(ticker, strength, price, reason, prediction_id=prediction_id)

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
        prediction_id: Optional[int] = None,
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
            #q 确保最后一手减掉
            if reduce_size == 0:
                reduce_size = 1
        else:
            # 默认减 30%
            reduce_size = int(pos.size * 0.3)

        if reduce_size <= 0:
            return
        val = self.get_ticker_value(ticker=ticker, price=price)
        if val <= 1000:
            reduce_size = pos.size  # 直接清仓
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
            prediction_id=prediction_id
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
            print(f'cash:{self.cash}')
            size = pos.size
            pos.size = 0
            self._record(
                symbol=ticker,
                action="CLOSE",
                size=size,
                price=price,
                value=value,
                reason=reason,
            )

    # ==================================================
    # Size(手) 计算（模拟 / 实盘统一入口）
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
        prediction_id: Optional[int] = None,
    ):
        self.max_occupied = max(self.max_occupied, self.get_ticker_value(symbol, price) )
        ts = self.current_market_time if self.current_market_time else datetime.now()

            # 动态增加计数
        self.total_trade_count += 1
        self.trade_log.append(
            {
                "ts": ts,
                "symbol": symbol,
                "action": action,
                "size": size,
                "price": price,
                "value": value,
                "reason": reason,
                "run_mode": GlobalState.mode,
            }
        )

        notify_order(
            OrderEvent(
                ts=ts,
                symbol=symbol,
                action=action,
                side="sell" if action in ["REDUCE", "CLOSE"] else "buy",
                size=size,
                price=price,
                value=value,
                reason=reason,
                prediction_id=prediction_id
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
                highest_price=p.get("highest_price", p["entry_price"]),             
            )

    def load_watchlist_from_csv(self, ticker_list: list):
        """新增方法：专门更新观察池，不碰真实持仓"""
        self.watchlist.clear() 
        for ticker in ticker_list:
            # 只有当该标的不在真实持仓中时，才加入观察池
            if ticker not in self.positions:
                self.watchlist[ticker] = {
                    "ticker": ticker,
                    "status": "WATCHING"
                }

    def clear(self):
        self.positions.clear()
        self.trade_log.clear()
        self.save(GlobalState.mode)  # 每次清仓后保存状态，确保实盘数据正确持久化
        
    def save(self,mode:RunMode):
       persist_live_positions(self,mode)