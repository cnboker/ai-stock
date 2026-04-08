
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import httpx
from http.client import HTTPException


# 2. 调用批量接口 (假设你已经定义了该异步函数)
# 注意：这里返回的是我们之前定义的 List[MarketDepth]
async def fetch_prices_as_dict(symbols: str):
    # 调用之前定义的获取接口
    batch_data = await get_batch_quotes(symbols)
    #print(f"batch_data={batch_data}")
    return parse_xueqiu_batch_to_dict(batch_data)
   
   

def parse_xueqiu_batch_to_dict(raw_json: Dict[str, Any]) -> Dict[str, float]:
    """
    将雪球批量接口返回的原始 JSON 转换为 {ticker: current_price} 字典
    """
    price_map = {}
    
    # 获取 items 列表
    items = raw_json.get("data", {}).get("items", [])
    
    for item in items:
        # 提取 quote 字段
        quote = item.get("quote")
        if not quote:
            continue
            
        # 提取股票代码和当前价格
        symbol = quote.get("symbol")
        current_price = quote.get("current")
        
        if symbol and current_price is not None:
            print(f"tiker={symbol}, price={current_price}")
            # 统一转为小写，确保与你的 task_queue 匹配
            price_map[symbol.lower()] = float(current_price)
            
    return price_map

#批量获取当前价格
async def get_batch_quotes(symbols: str):
    """
    symbols 格式: "SH600096,SZ000001"
    """
    # 雪球需要相关的 Cookie (xq_a_token)，建议从环境变量获取
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": "xq_a_token=c9d37c71ae91b784fb5573fef5ef372d4ba77e5a" 
    }
    
    url = f"https://stock.xueqiu.com/v5/stock/batch/quote.json?symbol={symbols}"
    print(f"url={url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            
            # 如果雪球返回 400/500，直接抛出 httpx 异常
            response.raise_for_status() 
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            print(f"❌ 雪球 API 状态码异常: {e.response.status_code}")
            return {"data": {"items": []}} # 返回空数据，防止上层崩溃
        except Exception as e:
            print(f"❌ 网络请求发生错误: {e}")
            return {"data": {"items": []}}


class OrderBookEntry(BaseModel):
    price: float = 0.0
    volume: int = 0

class MarketDepth(BaseModel):
    symbol: str
    timestamp: datetime
    current: float
    bids: List[OrderBookEntry]  # 买盘 [1-5]
    asks: List[OrderBookEntry]  # 卖盘 [1-5]
    buy_pct: float = 0.0
    sell_pct: float = 0.0
    diff: float = 0.0
    ratio: float = 0.0

def parse_xueqiu_item(quote: dict) -> MarketDepth:
    """
    解析雪球批量接口返回的单个 quote 字典
    """
    # 1. 处理时间戳 (雪球通常返回毫秒)
    ts = quote.get("timestamp")
    dt = datetime.fromtimestamp(ts / 1000.0) if ts else datetime.now()

    # 2. 提取买盘五档 (bids)
    # 逻辑：循环 1-5 档，若价格不存在则跳过，保证数据干净
    bids = []
    for i in range(1, 6):
        bp = quote.get(f"bp{i}")
        bc = quote.get(f"bc{i}")
        if bp is not None:
            bids.append(OrderBookEntry(price=float(bp), volume=int(bc or 0)))

    # 3. 提取卖盘五档 (asks)
    asks = []
    for i in range(1, 6):
        sp = quote.get(f"sp{i}")
        sc = quote.get(f"sc{i}")
        if sp is not None:
            asks.append(OrderBookEntry(price=float(sp), volume=int(sc or 0)))

    # 4. 构建并返回结构化对象
    return MarketDepth(
        symbol=quote.get("symbol", "UNKNOWN"),
        timestamp=dt,
        current=float(quote.get("current") or 0.0),
        bids=bids,
        asks=asks,
        buy_pct=float(quote.get("buypct") or 0.0),
        sell_pct=float(quote.get("sellpct") or 0.0),
        diff=float(quote.get("diff") or 0.0),
        ratio=float(quote.get("ratio") or 0.0)
    )