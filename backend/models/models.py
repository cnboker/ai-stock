from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
from pydantic import field_validator
import numpy as np

class Order(SQLModel, table=True):
    __tablename__ = "orders"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    side: str                                 # "buy" 或 "sell"
    quantity: int                             
    price: float                              
    entry_time: datetime = Field(index=True)
    
    # 盈亏计算核心
    realized_pnl: float = Field(default=0.0)  
    prediction_id: Optional[int] = Field(default=None, foreign_key="predictions.id")

    @field_validator("entry_time", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            # 移除 ISO 格式中可能存在的 Z 或空格
            v = v.replace("Z", "").replace(" ", "T")
            return datetime.fromisoformat(v)
        return v

    @field_validator("price", "realized_pnl", mode="before")
    @classmethod
    def handle_numpy_float(cls, v):
        # 自动修复 numpy.float64 导致的 JSON/DB 写入错误
        if isinstance(v, (np.float64, np.float32)):
            return float(v)
        return v

    @field_validator("quantity", mode="before")
    @classmethod
    def handle_numpy_int(cls, v):
        # 自动修复 numpy.int64 错误
        if isinstance(v, (np.int64, np.int32)):
            return int(v)
        return v
class Position(SQLModel, table=True):
    """用于实时维护每只股票的持仓成本"""
    __tablename__ = "positions"
    
    symbol: str = Field(primary_key=True)
    quantity: int = Field(default=0)          # 当前持股数
    avg_price: float = Field(default=0.0)     # 摊薄持仓成本价
    updated_at: datetime = Field(default_factory=datetime.now)

# Prediction 模型保持你提供的结构
class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    expected_return: float
    features_snapshot: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = "chronos-v1"