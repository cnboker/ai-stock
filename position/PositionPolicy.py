from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionAction:
    action: str                 # OPEN / REDUCE / EXIT / HOLD
    size: int                   # 本次变动的手数
    contract_size: int = 100
    plan: Optional[object] = None 
    ratio: float = 0.3 #减仓比例
class PositionPolicy:
    def __init__(
        self,
        score_threshold: float,
        max_reject_before_exit: int = 3,
        reduce_step: float = 0.3,
    ):
        self.score_threshold = score_threshold
        self.max_reject_before_exit = max_reject_before_exit
        self.reduce_step = reduce_step

    def decide(self, position, gate) -> PositionAction:
        """
        position: Position | None
        gate: GateResult
        """

        # 没仓位 → 永远 HOLD
        if position is None or position.size == 0:
            return PositionAction("HOLD")

        # Gate 允许 → 不干预
        if gate.allow:
            position.gate_reject_count = 0
            return PositionAction("HOLD")

        # Gate Reject
        position.gate_reject_count += 1

        # 达到清仓阈值
        if position.gate_reject_count >= self.max_reject_before_exit:
            return PositionAction("CLOSE")

        # 否则：渐进减仓
        return PositionAction(
            action="REDUCE",
            ratio=self.reduce_step,
        )
    
position_policy = PositionPolicy(
    score_threshold=0.6,
    max_reject_before_exit=3,
    reduce_step=0.3,
)