# snapshot.py
from dataclasses import dataclass

@dataclass
class DecisionSnapshot:
    timestamp: str
    ticker: str
    price: float
    action: str  # BUY / SELL / HOLD
    position: float
    equity: float
    notes: str = ""
