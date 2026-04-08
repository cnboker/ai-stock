
import os
import time
import json
import re

import pandas as pd
import requests
from diskcache import Cache
from infra.core.runtime import RunMode,GlobalState

# 初始化缓存目录（在项目根目录创建 .cache 文件夹）
cache_dir = os.path.join(os.getcwd(), ".cache_data")
cache = Cache(cache_dir)



def load_stock_df(ticker: str, period: str, expire_seconds: int = 3600) -> pd.DataFrame:
    """
    加载个股 K 线（带磁盘缓存功能）
    :param ticker: 股票代码
    :param period: 时间范围
    :param expire_seconds: 缓存失效时间（秒），默认 1 小时
    """
    # 生成唯一的缓存 Key
    cache_key = f"stock_data_{ticker}_{period}"
    
    # 尝试从缓存获取
    cached_df = cache.get(cache_key)
    
    if cached_df is not None and GlobalState.mode != RunMode.LIVE:
        # print(f"--- [Cache Hit] {ticker} ---")
        return cached_df

    # --- [Cache Miss] 执行原有的下载逻辑 ---
    # print(f"--- [Downloading] {ticker} from network ---")
    
    # 这里的 download 替换为你实际的下载函数
    df = download(ticker, period=period) 
    
    if df is None or df.empty:
        raise ValueError(f"{ticker} 行情为空")
        
    # 数据清洗
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    
    # 存入缓存，设置过期时间
    cache.set(cache_key, df, expire=expire_seconds)
    
    return df

def load_index_df(period: str) -> pd.DataFrame:
    """
    加载沪深300指数数据，用于条件化预测
    """
    return load_stock_df("sh000300",period) 
  

def fetch_sina_quote(symbol,minites):
    #datalen=512(30min, 3months)
    ts = int(time.time() * 1000)  # 毫秒时间戳强制避免缓存
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_{minites}_{ts}=/CN_MarketDataService.getKLineData?symbol={symbol}&scale={minites}&ma=no&datalen=1024"
    #print('url->',url)
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.encoding = "gbk"  # 新浪编码
        return r.text
    except Exception as e:
        print("请求失败：", e)
        return None
    

def download(symbol, period):
    raw = fetch_sina_quote(symbol,period)
 
    if not raw:
        return    
    return convert_to_df(raw=raw)


def convert_to_df(raw):
    # 去掉 var xxx = 和最后的 );
    m = re.search(r'=\s*\((.*)\)\s*;', raw, re.S)
    if not m:
        return pd.DataFrame()

    array_str = m.group(1)

    # 替换单引号为双引号才能 json.loads
    array_str = array_str.replace("'", '"')
   
    try:
        arr = json.loads(array_str)
    except:
        return pd.DataFrame()

    # 转 DataFrame
    df = pd.DataFrame(arr)

    # 转换字段类型
    df["day"] = pd.to_datetime(df["day"])
    df = df.set_index("day")

    float_cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

import re

def extract_latest_price(parts: list[str]) -> float:
    """
    parts: ['金证股份', '15.600', ..., '15.860', '2025-12-25', '13:31:35', '00', '']
    返回最新价格 float
    """
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")  # 匹配日期 YYYY-MM-DD
    
    for i in range(len(parts)-1, -1, -1):
        if date_pattern.match(parts[i]):
            # 日期前一个就是最新价格
            try:
                latest_price = float(parts[i-1])
                return latest_price
            except (ValueError, IndexError):
                return 0.0
    # 没找到日期，返回0
    return 0.0

def parse_multiple_latest_prices(raw: str) -> dict[str, float]:
    price_map = {}
    
    # 匹配每个 var hq_str_ticker="内容";
    matches = re.findall(r'var hq_str_(\w+)="([^"]+)"', raw)
    
    for ticker, content in matches:
        parts = content.split(',')
        latest_price = extract_latest_price(parts)
        
        if latest_price is not None:
            price_map[ticker] = latest_price
        else:
            price_map[ticker] = 0.0
    
    return price_map




def fetch_sina_quote_live(symbols):
    now = time.time()

    # 发送请求
    ts_ms = int(now * 1000)  # 毫秒时间戳
    url = f"https://hq.sinajs.cn/etag.php?_={ts_ms}&list={','.join(symbols)}"
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.encoding = "gbk"  # 新浪编码
        # print(r.text)
        data = parse_multiple_latest_prices(r.text)
    
        return data
    except Exception as e:
        print("请求失败：", e)
        return None