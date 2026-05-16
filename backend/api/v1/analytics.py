from models.models import Position, Prediction, Order
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from api.deps import get_session  # 假设你在 session.py 定义了 get_session

router = APIRouter()


@router.post("/predictions", response_model=Prediction)
async def create_prediction(
    pred: Prediction, session: AsyncSession = Depends(get_session)
):
    # 强制检查并转换时间格式（防止驱动报错）
    if isinstance(pred.timestamp, str):
        try:
            # 将字符串转换为 datetime 对象
            pred.timestamp = datetime.fromisoformat(pred.timestamp.replace("Z", ""))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use ISO format."
            )
    """记录一次 Chronos 预测信号"""
    session.add(pred)
    await session.commit()
    await session.refresh(pred)
    return pred


@router.get("/predictions", response_model=List[Prediction])
async def list_predictions(
    symbol: Optional[str] = None,
    date: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """查询预测记录"""
    statement = select(Prediction)
    if symbol:
        statement = statement.where(Prediction.symbol == symbol)
    if date:
        statement = statement.where(func.date(Prediction.timestamp) == date)

    return session.exec(statement).all()


@router.post("/orders", response_model=Order)
async def execute_order(order_obj: Order, session: AsyncSession = Depends(get_session)):
    """
    处理订单并自动计算 PnL 和更新持仓成本
    """
    order_data = Order.model_validate(order_obj)
    # 1. 强制获取 Position 实例
    pos = await session.get(Position, order_data.symbol)

    if order_data.side == "sell":
    # 找到该标的的持仓
        pos = await session.get(Position, order_data.symbol)
        if pos:
            # 计算 PnL 存入 Order 表
            order_data.realized_pnl = (order_data.price - pos.avg_price) * order_data.quantity
            # 更新持仓
            pos.quantity -= order_data.quantity
            session.add(pos)

    elif order_data.side == "buy":
        if pos:
            # 加权平均成本计算
            new_total_qty = pos.quantity + order_data.quantity
            pos.avg_price = (
                (pos.avg_price * pos.quantity)
                + (order_data.price * order_data.quantity)
            ) / new_total_qty
            pos.quantity = new_total_qty
        else:
            # 首次建仓
            pos = Position(
                symbol=order_data.symbol,
                quantity=order_data.quantity,
                avg_price=order_data.price,
            )

        session.add(pos)

    # 6. 保存订单
    session.add(order_data)
    await session.commit()
    await session.refresh(order_data)
    return order_data


@router.patch("/orders/{order_id}", response_model=Order)
async def update_order_status(
    order_id: int,
    actual_return: float,
    pnl: float,
    session: AsyncSession = Depends(get_session),
):
    """更新订单结果（通常在收盘后用于填入实际收益）"""
    db_order = session.get(Order, order_id)
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")

    db_order.actual_return = actual_return
    db_order.pnl_amount = pnl
    db_order.status = "closed"

    session.add(db_order)
    await session.commit()
    await session.refresh(db_order)
    return db_order


@router.get("/daily_review/{date}")
async def get_daily_review(date: str, session: AsyncSession = Depends(get_session)):
    target_date = datetime.strptime(date, "%Y-%m-%d").date()

    # 1. 获取当天的所有预测
    pred_stmt = select(Prediction).where(func.date(Prediction.timestamp) == target_date)
    result = await session.execute(pred_stmt)
    preds = result.scalars().all()
    review_data = []
    total_pnl = 0.0

    for pred in preds:
        # 2. 寻找该预测之后，针对该标的的成交记录（包含减仓记录）
        order_stmt = select(Order).where(Order.prediction_id == pred.id)

        result = await session.execute(order_stmt)
        orders = result.scalars().all()

        # 汇总该预测带来的所有已实现盈亏
        actual_pnl = sum(o.realized_pnl for o in orders)
        total_pnl += actual_pnl

        # 计算误差：这里可以用 (实际收益率 - 预期收益率)
        # 实际收益率 = 实际盈亏 / (预测时的成本 * 数量)
        review_data.append(
            {
                "symbol": pred.symbol,
                "prediction_id": pred.id,  # 建议加上 ID 方便对账
                "expected_return": pred.expected_return,
                "actual_pnl": actual_pnl,
                "features_snapshot": pred.features_snapshot,
                "pred_time": pred.timestamp,                
                "order_count": len(orders),
                "status": (
                    "Partially Closed"
                    if any(o.side == "sell" for o in orders)
                    else "Holding"
                ),
            }
        )

    return {
        "date": target_date,
        "summary": {"total_pnl": round(total_pnl, 2), "prediction_count": len(preds)},
        "details": review_data #[data for data in review_data if data["order_count"] > 0],  # 只展示有成交记录的预测 
    }
