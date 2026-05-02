import asyncio
import os
import re
import json
from typing import Dict, List
from diskcache import Cache
import httpx
from matplotlib import ticker

from infra.core.runtime import GlobalState, RunMode

cache_dir = os.path.join(os.getcwd(), ".cache_data")
cache = Cache(cache_dir)


# 原始数据字符串
raw_data = 'quotebridge_v6_time_hs_688256_defer_last({"hs_688256":{"name":"\u5bd2\u6b66\u7eaa","open":0,"stop":0,"isTrading":0,"rt":"0930-1130,1300-1500,1505-1530","tradeTime":["0930-1130","1300-1500","1505-1530"],"pre":"1416.63","date":"20260430","data":"0930,1475.00,321293350,1475.000,217826;0931,1487.99,910686940,1480.333,614406;0932,1491.19,606740510,1482.282,408234;0933,1499.95,491805900,"}})'

data_cache = {}
base_url = "https://d.10jqka.com.cn/v6/time/"
headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36"}


# 获取前一天的价格数据，构建 {时间: 价格} 的字典
async def get_last_day_price(
    symbol: str, timestamp: str = None, expire_seconds: int = 3600
) -> float:
    symbol = format_ticker(symbol)
    cache_key = f"stock_data_{ticker}_last_price"

    # 尝试从缓存获取
    cached_df = cache.get(cache_key)

    if cached_df is not None and GlobalState.mode != RunMode.LIVE:
        # print(f"--- [Cache Hit] {ticker} ---")
        price_map = cached_df
    else:
        price_map = await get_price_map(symbol)
        # print(f"获取到的价格数据: {price_map}")
        # 存入缓存，设置过期时间
        cache.set(cache_key, price_map, expire=expire_seconds)
    """
    获取指定时间点的价格，如果 timestamp 为空则获取最新价格
    """ 
    if timestamp:
        return price_map.get(timestamp, 0.0)  # 返回指定时间的价格，找不到则返回0
    else:
        # 返回最新价格，即最后一个时间点的价格
        if price_map:
            latest_time = max(price_map.keys())
            return price_map[latest_time]
        else:
            return 0.0


async def get_price_map(symbol: str) -> Dict[str, float]:
    """
    返回精简的价格字典，例如: {'sz300961': 12.77, 'sh600519': 1650.0}
    """
    symbol = format_ticker(symbol)
    if symbol in data_cache:
        return data_cache[symbol]

    # 统一转为小写处理

    query_str = f"{symbol}/defer/last.js"
    url = f"{base_url}{query_str}"
    # print(f"url={url}")
    proxy_url = "http://127.0.0.1:7890"

    async with httpx.AsyncClient(proxy=proxy_url, trust_env=True) as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code != 200:
                return {}
            # print(response.text)
            return extract_price_map(response.text)
        except Exception as e:
            print(f"获取行情失败: {e}")
            return {}


def format_ticker(ticker):
    """将 hs688256 转换为 hs_688256"""
    # 使用正则在字母和数字边界插入下划线
    return re.sub(r"([a-zA-Z]+)(\d+)", r"\1_\2", ticker)


def extract_price_map(data_str):
    # 1. 提取括号内的 JSON 内容
    content = re.search(r"\((.*)\)", data_str).group(1)
    json_data = json.loads(content)

    # 2. 获取嵌套的 data 字符串 (以 "hs_688256" 为例)
    # 这里通过 next(iter(...)) 自动适配不同的股票代码 key
    ticker_key = next(iter(json_data))
    raw_time_series = json_data[ticker_key]["data"]

    # 3. 解析 data 并构建 map
    # 格式为: 时间,价格,成交额,均价,成交量;...
    price_map = {}
    segments = raw_time_series.split(";")

    for seg in segments:
        if not seg:
            continue
        parts = seg.split(",")
        if len(parts) >= 2:
            time_str = parts[0]
            price = float(parts[1])
            price_map[time_str] = price

    return price_map


# --- 使用示例 ---
async def main():
    result = await get_last_day_price("hs688256", "0931")
    print(result)


# if __name__ == "__main__":
#     asyncio.run(main())
