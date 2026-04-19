import httpx
import asyncio
import json
from datetime import datetime

API_BASE = "http://localhost:8080/api/v1/analytics"

async def upload_data():
    async with httpx.AsyncClient() as client:
        # --- 步骤 1: 构造并上传 Prediction (预测数据) ---
        # 对应日志中的 "交易前" 和 Chronos 输出部分
        prediction_payload = {
            "symbol": "sh512760",
            "timestamp": "2026-04-19T13:30:00",
            "expected_return": 0.005088,  # 对应日志中的 predicted_up
            "confidence_interval_low": 0.7164, # 对应 raw_score
            "confidence_interval_high": 0.7652, # 对应 model_score
            "features_snapshot": json.dumps({
                "regime": "good",
                "gate_mult": 0.9363,
                "confidence": 0.7652,
                "atr": 0.0145
            }),
            "model_version": "chronos-v1"
        }
        
        pred_resp = await client.post(f"{API_BASE}/predictions", json=prediction_payload)
        print(f"✅ Prediction uploaded, ID: {pred_resp}")
        pred_data = pred_resp.json()
        prediction_id = pred_data.get("id")
        print(f"✅ Prediction uploaded, ID: {prediction_id}")

        # --- 步骤 2: 构造并上传 Order (订单结果) ---
        # 对应日志中的开仓和止损平仓数据
        # 实际收益率计算: (0.79 - 1.59) / 1.59 = -0.5031
        # 盈亏金额: 19728.60 - 39909.00 = -20180.4
        
        order_payload = {
            "symbol": "sh512760",
            "side": "buy",
            "entry_price": 1.59,
            "exit_price": 0.79,
            "entry_time": "2026-04-19T9:30:00",
            "exit_time": "2026-04-19T11:00:00",
            "actual_return": -0.5031,
            "pnl_amount": -20180.4,
            "prediction_id": prediction_id, # 建立关联！
            "status": "closed"
        }
        
        order_resp = await client.post(f"{API_BASE}/orders", json=order_payload)
        if order_resp.status_code == 200:
            print(f"✅ Order uploaded successfully for {order_payload['symbol']}")
        else:
            print(f"❌ Order upload failed: {order_resp.text}")

if __name__ == "__main__":
    asyncio.run(upload_data())