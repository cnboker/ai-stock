# core/context.py
from dataclasses import dataclass, field
from typing import Optional
from infra.core.runtime import RunMode
from equity.equity_features import equity_features
from position.position_manager import PositionManager
from equity.equity_recorder import EquityRecorder
from pandas import DataFrame
from config.settings import ticker_name_map

@dataclass
class TradingContext:
    run_mode: RunMode
    position_mgr: PositionManager
    eq_recorder: EquityRecorder
    ticker: str
    period: str
    hs300_df: DataFrame
    name: Optional[str]= field(default=None)
    eq_feat: Optional[DataFrame] = field(default=None)

    def __post_init__(self):
        # 自动计算 eq_feat，如果没有传入
        self.name = ticker_name_map.get(self.ticker, self.ticker)
        if self.eq_feat is None and self.eq_recorder is not None:
            self.eq_feat = equity_features(self.eq_recorder.to_series())

