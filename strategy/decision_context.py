from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json

class GoodHoldReason(Enum):
    NONE = auto()

    # HARD / SUPPRESSED
    COOLDOWN_ACTIVE = auto()
    CONFIRMATION_PENDING = auto()
    POSITION_LIMIT = auto()

    # SOFT / SIGNAL
    GATE_NOT_PASSED = auto()
    STRENGTH_TOO_WEAK = auto()
    TREND_DECAY = auto()


@dataclass(frozen=True)
class DecisionContext:
    # ===== 标识 =====
    ticker: str

    # ===== 市场 =====
    latest_price: float
    atr: float

    # ===== 模型 =====
    model_score: float
    predicted_up: float

    # ===== Gate / 信号 =====
    gate_allow: bool
    gate_mult: float
    slope: float
    strength: float

    # ===== 资金 / 风控 =====
    regime: str
    regime_cooldown_left: float
    dd: float

    # ===== 仓位 =====
    has_position: bool
    allow_add: bool
    position_size: float

    # ===== regime 确认 =====
    good_count: int
    good_confirm_need: int

    # ===== 原始信号 =====
    raw_signal: str
    raw_score: float

    # ===== 策略阈值（事实，不是逻辑）=====
    slope_decay_thresh: float = 0.0

    # =========================
    # JSON / Dict
    # =========================
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
