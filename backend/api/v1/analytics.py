from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func

from api.deps import get_session  # 假设你在 session.py 定义了 get_session
from models.prediction import Prediction
from models.order import Order

router = APIRouter()

@router.post("/predictions", response_model=Prediction)
async def create_prediction(pred: Prediction, session: Session = Depends(get_session)):
    """记录一次 Chronos 预测信号"""
    session.add(pred)
    session.commit()
    session.refresh(pred)
    return pred

@router.get("/predictions", response_model=List[Prediction])
async def list_predictions(
    symbol: Optional[str] = None, 
    date: Optional[str] = None,
    session: Session = Depends(get_session)
):
    """查询预测记录"""
    statement = select(Prediction)
    if symbol:
        statement = statement.where(Prediction.symbol == symbol)
    if date:
        statement = statement.where(func.date(Prediction.timestamp) == date)
    
    return session.exec(statement).all()


# --- Order 接口 ---
@router.post("/orders", response_model=Order)
async def create_order(order: Order, session: Session = Depends(get_session)):
    """记录一次实盘交易成交情况"""
    session.add(order)
    session.commit()
    session.refresh(order)
    return order

@router.patch("/orders/{order_id}", response_model=Order)
async def update_order_status(
    order_id: int, 
    actual_return: float, 
    pnl: float,
    session: Session = Depends(get_session)
):
    """更新订单结果（通常在收盘后用于填入实际收益）"""
    db_order = session.get(Order, order_id)
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    db_order.actual_return = actual_return
    db_order.pnl_amount = pnl
    db_order.status = "closed"
    
    session.add(db_order)
    session.commit()
    session.refresh(db_order)
    return db_order

@router.get("/daily_review")
async def get_daily_review(
    date: Optional[str] = None, 
    session: Session = Depends(get_session)
):
    """
    Week 1 核心接口：获取每日预测与实盘对比数据，供 Hermes 审计
    """
    # 1. 处理日期：默认查询当天
    target_date = date if date else datetime.now().strftime("%Y-%m-%d")
    
    try:
        # 2. 查询当天的预测数据
        # 注意：这里假设你的 timestamp 是 datetime 类型，我们需要 cast 或匹配日期部分
        pred_statement = select(Prediction).where(
            func.date(Prediction.timestamp) == target_date
        )
        predictions = session.exec(pred_statement).all()
        
        # 3. 查询当天的成交/订单数据
        order_statement = select(Order).where(
            func.date(Order.entry_time) == target_date
        )
        orders = session.exec(order_statement).all()
        
        # 4. 聚合逻辑：将预测与实际结果对齐
        review_data = []
        for pred in predictions:
            # 在当天的订单中找到对应 symbol 的结果
            match = next((o for o in orders if o.symbol == pred.symbol), None)
            
            actual_ret = match.actual_return if match else 0.0
            error = abs(pred.expected_return - actual_ret) if match else None
            
            review_data.append({
                "symbol": pred.symbol,
                "pred_change": round(pred.expected_return, 4),
                "actual_change": round(actual_ret, 4),
                "error": round(error, 4) if error is not None else None,
                "status": "matched" if match else "no_trade",
                # 这里的 features 可以让 Hermes 知道当时的决策背景
                "context": pred.features_snapshot 
            })
        
        # 5. 宏观背景（示例逻辑：你可以根据所有 symbol 的平均表现或外部 API 获取）
        avg_error = sum([d["error"] for d in review_data if d["error"]]) / max(len(review_data), 1)
        
        return {
            "date": target_date,
            "summary": {
                "total_signals": len(predictions),
                "avg_error": round(avg_error, 4),
                "market_sentiment": "High Volatility" if avg_error > 0.02 else "Stable"
            },
            "data": review_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")