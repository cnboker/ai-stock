# core/context.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from position.position_manager import PositionManager
from equity.equity_recorder import EquityRecorder
from strategy.trade_intent import TradeIntent

@dataclass
class TradingSession:
    period: str
    hs300_df: DataFrame   

    position_mgr: PositionManager
    eq_recorder: EquityRecorder
    eq_feat: Optional[DataFrame] = field(default=None)
    tradeIntent:Optional[TradeIntent] = field(default=None)
    prices_today: ndarray = field(default_factory=lambda: np.empty(0))
   
