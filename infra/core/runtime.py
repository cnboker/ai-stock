# core/runtime.py
from enum import Enum

class RunMode(str, Enum):
    LIVE = "live"
    SIM = "sim"
    BACKTEST = "backtest"

class GlobalState:
    # 默认值
    mode: RunMode = RunMode.LIVE