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
    tickers_price:dict = {}