# Trading Order Models

from pydantic import field_validator

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

    # --- 核心修复：自动解析字符串时间 ---
    @field_validator("entry_time", "exit_time",mode="before")
    def parse_datetime(cls, v):
        if isinstance(v, str):
            try:
                # 兼容 ISO 格式字符串：'2026-04-21T15:07:50.445439'
                return datetime.fromisoformat(v)
            except ValueError:
                # 如果带 Z 或者其他格式，可以用更通用的解析
                return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f")
        return v