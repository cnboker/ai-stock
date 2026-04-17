# Trading Order Models
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Order(SQLModel, table=True):
    __tablename__ = "orders"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    side: str                                 # "buy" 或 "sell"
    
    # 价格相关
    entry_price: float                        # 入场价格
    exit_price: Optional[float]               # 出场价格
    entry_time: datetime = Field(index=True)
    exit_time: Optional[datetime]
    
    # 结果相关 (Week 1 复盘核心)
    actual_return: float = Field(default=0.0) # 实际收益率
    pnl_amount: float = Field(default=0.0)    # 实际盈亏金额
    
    # 关联字段
    prediction_id: Optional[int] = Field(foreign_key="predictions.id") # 关联到预测ID
    status: str = "closed"                    # "open", "closed", "cancelled"