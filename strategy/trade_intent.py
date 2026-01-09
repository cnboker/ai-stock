from dataclasses import dataclass, field

from strategy.decision_context import GoodHoldReason

#我打算做什么
@dataclass
class TradeIntent:    
    #===== 决策结果层 =====
    action:str                # ->最终执行动作,decide_equity_policy->cooldown_mgr.update后， cooldown 之后的最终动作
    raw_action: str | None = None  # 原始意图（新增）;decide_equity_policy执行后  

    ## ===== 市场 / 风控状态 =====
    regime: str | None = None             # good / neutral / bad 资金/->风险状态     
    gate_mult: float | None = None        # 仓位放大/压制 ->风控调制
    reduce_strength: float | None = None  # 0~1 执行参数
    force_reduce: bool = False      # 是否 强制减仓 ->强制约束
    gate_allow:bool = False

    # ===== 决策上下文（ctx）=====
    model_score: float = 0.0
    raw_score: float = 0.0      # 连续 score（模型 × gate × equity）->模型派生
    predicted_up: float = 0.0
    confidence: float = 0.0     # 事件强度（来自 debouncer）->信号质量
    confirmed: bool = False    # 是否通过 debouncer ->稳定器结果
    atr: float = 0.0
    strength: float = 0.0            # compute_strength

    has_position: bool = False
    cooldown_active: bool = False
    cooldown_left: int = 0

    # ===== 解释层（新增）=====
    good_hold_reason: GoodHoldReason = GoodHoldReason.NONE
    good_hold_detail: dict = field(default_factory=dict)

    # ===== 通用日志原因（保留）=====
    reason: str = ""            # 触发原因（日志 / 回测用）



