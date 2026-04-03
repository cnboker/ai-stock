from dataclasses import dataclass
from typing import Optional
from datetime import datetime
@dataclass
class Position:
    ticker: str
    direction: Optional[str] = None   # LONG / SHORT
    entry_price: float = 0.0 #持仓成本，加权平均价，考虑加仓后的价格变化

    size: int = 0
    stop_loss: float = 0.0    
    open_time:datetime = datetime.now()    
    contract_size:int = 100
    #用于 trailing stop
    highest_price: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
     # 状态机
    stage = "init"   # init → profit_lock → trend
    