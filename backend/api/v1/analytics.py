# Week 1: Hermes Review APIs

router = APIRouter()
@app.get("/api/v1/daily_review")
async def get_daily_review(date: str = None):
    # date 格式: 2026-04-15
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 1. 从数据库读取当天的信号预测数据
    predictions = db.query(Predictions).filter(Predictions.date == date).all()
    # 2. 从数据库读取当天的实际收盘价及 PnL
    actuals = db.query(Orders).filter(Orders.date == date).all()
    
    # 3. 聚合逻辑：对比预测与实际
    review_data = []
    for pred in predictions:
        match = next((a for a in actuals if a.symbol == pred.symbol), None)
        review_data.append({
            "symbol": pred.symbol,
            "pred_change": pred.expected_return, # Chronos 预测值
            "actual_change": match.actual_return if match else 0,
            "error": abs(pred.expected_return - match.actual_return) if match else None,
            "volume_status": match.volume_ratio # 结合量能看是否有环境偏差
        })
    
    return {
        "date": date,
        "market_summary": "缩量上涨" if is_low_volume else "放量下跌", # 示例宏观背景
        "data": review_data
    }