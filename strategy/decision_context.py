
from dataclasses import dataclass

@dataclass
class DecisionContext:
    # 1. 基础信息 (无默认值)
    ticker: str
    latest_price: float
    atr: float
    
    # 2. 模型预测 (无默认值)
    model_score: float # 模型原始分
    predicted_up: float # 预期涨幅
    
    # 3. 结构与风控系数 (无默认值)
    gate_allow: bool # Gater 是否放行
    gate_mult: float   # 综合缩放因子 (Gater * EquityRisk)    
    
    # 4. 状态判定 (无默认值)
    regime: str       # bad/good/neutral     
    has_position: bool
    position_size: float
    
    # 5. 信号指令 (无默认值)
    raw_signal: str        # LONG/HOLD/REDUCE/LIQUIDATE
    raw_score: float       
    reduce_strength: float 
    
    # 6. 过滤参数 (无默认值)
    slope: float           # <--- 移动到这里，因为它没有默认值
    strength: float  # <--- 加在这里，没有默认值
    # 7. 带有默认值的参数 (必须放在最后)
    liquidate_reason: str = None