from dataclasses import dataclass
from typing import Optional
from datetime import datetime
@dataclass
class Position:
    ticker: str
    direction: Optional[str] = None   # LONG / SHORT
    entry_price: float = 0.0
    size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    open_time:datetime = datetime.now()