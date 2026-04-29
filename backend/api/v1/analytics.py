from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from api.deps import get_session  # 假设你在 session.py 定义了 get_session
from models.prediction import Prediction
from models.order import Order

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


# --- Order 接口 ---
@router.post("/orders", response_model=Order)
async def create_order(order: Order, session: AsyncSession = Depends(get_session)):
    """记录一次实盘交易成交情况"""
    db_order = Order.model_validate(order)
    session.add(db_order)
    await session.commit()
    await session.refresh(db_order)
    return db_order


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


@router.get("/daily_review")
async def get_daily_review(
    date: Optional[str] = None, session: AsyncSession = Depends(get_session)
):
    """
    Week 1 核心接口：获取每日预测与实盘对比数据，供 Hermes 审计
    """
    # 1. 处理日期：默认查询当天
    target_date = datetime.now().date()
    try:
        # 2. 查询当天的预测数据
        pred_statement = select(Prediction).where(
            func.date(Prediction.timestamp) == target_date
        )
        results = await session.exec(pred_statement)
        predictions = results.all()
        # 3. 查询当天的成交/订单数据
        order_statement = select(Order).where(
            func.date(Order.entry_time) == target_date
        )
        result_1 = await session.exec(order_statement)
        orders = result_1.all()

        review_data = generate_review_data(predictions, orders)
        avg_error = sum([d["error"] for d in review_data if d["error"]]) / max(
            len(review_data), 1
        )

        return {
            "date": target_date,
            "summary": {
                "total_signals": len(predictions),
                "avg_error": round(avg_error, 4),
                "market_sentiment": "High Volatility" if avg_error > 0.02 else "Stable",
            },
            "data": review_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


def generate_review_data(predictions, orders):
    """
    predictions: 预测信号列表，包含 symbol, expected_return, timestamp, features_snapshot
    orders: 数据库订单列表，包含 symbol, actual_return, pnl_amount, type(OPEN/REDUCE), finish_time
    """
    review_data = []

    # 1. 预过滤：只保留有实际收益贡献的减仓单/平仓单
    # 排除 OPEN 单，因为 OPEN 单的 actual_return 通常为 0，会拉低均值
    realized_orders = [o for o in orders if o.status in ["REDUCE", "CLOSE", "SELL"]]

    # 2. 对预测信号按时间排序（确保逻辑线性）
    sorted_preds = sorted(predictions, key=lambda x: x.timestamp)

    # 确保所有时间戳都是 datetime 对象且包含毫秒
    def to_dt(ts):
        if isinstance(ts, str):
            # 兼容带毫秒和不带毫秒的格式
            try:
                return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
            except:
                return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return ts

    for i, pred in enumerate(sorted_preds):
        curr_pred_time = to_dt(pred.timestamp)

        # 优化点 1：结束时间稍微往后偏移一点（比如 1ms），或者包含下一个信号的起始点
        # 或者干脆定义：这个信号负责到它“之后”所有该 symbol 的成交，直到下一个该 symbol 的信号出现

        # 找出该 symbol 之后的下一个信号时间
        next_same_symbol_pred = next(
            (p for p in sorted_preds[i + 1 :] if p.symbol == pred.symbol), None
        )
        end_time = (
            to_dt(next_same_symbol_pred.timestamp)
            if next_same_symbol_pred
            else datetime.max
        )

        # 优化点 2：容错性匹配
        # 交易执行往往比信号慢几毫秒，所以用 >= 是对的
        matches = [
            o
            for o in realized_orders
            if o.symbol == pred.symbol
            and curr_pred_time <= to_dt(o.exit_time) < end_time
        ]

        if matches:
            # 5. 聚合计算该窗口内的表现
            # 计算加权平均收益（如果所有减仓权重一致，可用算术平均）
            avg_actual_ret = sum(o.actual_return for o in matches) / len(matches)
            total_pnl = sum(o.pnl_amount for o in matches)

            error = abs(pred.expected_return - avg_actual_ret)
            status = "matched"
            order_count = len(matches)
        else:
            # 如果该信号发出后没有对应的操作
            avg_actual_ret = 0.0
            total_pnl = 0.0
            error = None  # 或者保留 abs(pred.expected_return)
            status = "no_trade"
            order_count = 0

        # 6. 构造输出数据
        review_data.append(
            {
                "symbol": pred.symbol,
                "signal_time": pred.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "pred_change": round(pred.expected_return, 4),
                "actual_change": round(avg_actual_ret, 4),
                "total_pnl": round(total_pnl, 2),
                "order_count": order_count,
                "error": round(error, 4) if error is not None else None,
                "status": status,
                "context": pred.features_snapshot,
            }
        )

    return review_data
