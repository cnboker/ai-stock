import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

# 建议在 settings 中配置 API 地址
API_BASE_URL = "http://127.0.0.1:8080/api/v1" 

def api_save_prediction(data: Dict[str, Any]) -> Optional[int]:
    """
    将 TradeIntent 预测意图存入数据库
    返回 prediction_id (int) 或 None
    """
    url = f"{API_BASE_URL}/analytics/predictions/"
    try:
        # 处理 datetime 序列化
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
            
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 201 or response.status_code == 200:
            return response.json().get("id")
        else:
            print(f"❌ Prediction API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Prediction API Connection Failed: {e}")
        return None

def api_save_order(data: Dict[str, Any]) -> bool:
    """
    将实际成交的 Order 存入数据库
    """
    url = f"{API_BASE_URL}/analytics/orders/"
    try:
        # 转换 datetime
        for key in ["entry_time", "exit_time"]:
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()

        response = requests.post(url, json=data, timeout=5)
        print(f"📤 Order API Response: {response.status_code} - {response.text} ")
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"❌ Order API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"⚠️ Order API Connection Failed: {e}")
        return False