# Chronos Prediction Models
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)           # 股票代码
    timestamp: datetime = Field(default_factory=datetime.now, index=True) # 预测时间
    
    # Chronos 核心输出
    expected_return: float                    # 预期收益率 (例如 0.05 代表 5%)
    confidence_interval_low: Optional[float]  # 置信区间下限
    confidence_interval_high: Optional[float] # 置信区间上限
    
    # 预测时的环境状态（交给 Hermes 分析误差原因的关键）
    features_snapshot: str                    # 存储当时的量能、RSI等(JSON字符串)
    model_version: str = "chronos-v1"         # 模型版本