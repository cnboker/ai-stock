# core/runtime.py
from enum import Enum

class RunMode(str, Enum):
    LIVE = "live"
    SIM = "sim"
    BACKTEST = "backtest"