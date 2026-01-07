# core/context.py
from dataclasses import dataclass, field
from typing import Optional
from infra.core.runtime import RunMode
from pandas import DataFrame

from strategy.equity_policy import TradeIntent

@dataclass
class TradingSession:
    run_mode: RunMode
    period: str
    hs300_df: DataFrame    
    eq_feat: Optional[DataFrame] = field(default=None)
    tradeIntent:Optional[TradeIntent] = field(default=None)
