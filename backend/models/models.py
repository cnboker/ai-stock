from sqlmodel import SQLModel, Field, select
from datetime import datetime
from typing import Optional, List
from pydantic import field_validator

class Order(SQLModel, table=True):
    __tablename__ = "orders"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    side: str                                 # "buy" (建仓/增仓) 或 "sell" (减仓/清仓)
    quantity: int                             # 成交数量
    price: float                              # 成交价格
    entry_time: datetime = Field(index=True)
    
    # 核心字段：本次操作产生的已实现盈亏
    # 如果是 buy，则为 0；如果是 sell，记录 (卖出价 - 之前均价) * 卖出数量
    realized_pnl: float = Field(default=0.0)  
    
    prediction_id: Optional[int] = Field(foreign_key="predictions.id")

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
    timestamp: datetime = Field(default_factory=datetime.now)