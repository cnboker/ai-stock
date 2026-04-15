# Week 4: Order Confirmation APIs
# trading 负责信号的拦截与确认
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

class TradeSignal(BaseModel):
    symbol: str
    expected_return: float
    confidence: float
    features: Dict

@router.post("/confirm_signal")
async def confirm_signal(signal: TradeSignal):
    """
    信号拦截器：在真正进入券商接口前，由 Hermes 进行二次语义确认
    """
    # 1. 这里未来会调用 app/services/hermes_agent.py 
    # 让 Hermes 基于“记忆库”判断这个信号是否可靠
    
    # 模拟 Hermes 的判断逻辑
    is_approved = True 
    reason = "符合近期高胜率形态"
    
    if signal.confidence < 0.6:
        is_approved = False
        reason = "置信度不足，Hermes 建议放弃"

    if not is_approved:
        return {
            "action": "REJECT",
            "reason": reason,
            "symbol": signal.symbol
        }

    return {
        "action": "EXECUTE",
        "reason": reason,
        "symbol": signal.symbol,
        "recommended_position": 0.5 # 建议半仓
    }