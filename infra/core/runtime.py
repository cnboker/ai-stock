# core/runtime.py
from enum import Enum

class RunMode(str, Enum):
    #实盘
    LIVE = "live"
    #回测
    SIM = "sim"


class GlobalState:
    # 默认值
    mode: RunMode = RunMode.LIVE
    # Chronos 需要的上下文长度（即 LOOKBACK_WINDOW）
    chronos_context_length:int = 20
    #回测窗口
    strategy_window:int = 20
    tickers_price:dict