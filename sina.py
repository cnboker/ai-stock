import json
import re
import pandas as pd
import requests
import time
import datetime

def fetch_sina_quote(symbols):
    """
    symbols: ["sz300697", "sh600000"]
    """
    ts = int(time.time() * 1000)  # 毫秒时间戳强制避免缓存
    
    url = f"https://hq.sinajs.cn/etag.php?_={ts}&list={','.join(symbols)}"
    
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



def fetch_sina_quote(symbol,minites):
    """
    symbols: ["sz300697", "sh600000"]
    """
    ts = int(time.time() * 1000)  # 毫秒时间戳强制避免缓存
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_{minites}_{ts}=/CN_MarketDataService.getKLineData?symbol={symbol}&scale={minites}&ma=no&datalen=1023"
    
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
    
def fetch_sina_quote_5m(symbol):
    """
    symbols: ["sz300697", "sh600000"]
    """
    ts = int(time.time() * 1000)  # 毫秒时间戳强制避免缓存
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_5_{ts}=/CN_MarketDataService.getKLineData?symbol={symbol}&scale=5&ma=no&datalen=1023"
    
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
        

def parse_sina_single(raw_line):
    """
    解析单只股票的返回字段
    """
    try:
        name = raw_line.split('"')[1]
        parts = raw_line.split('"')[1].split(',')
        
        return {
            "name": parts[0],
            "open": float(parts[1]),
            "yesterday_close": float(parts[2]),
            "price": float(parts[3]),
            "high": float(parts[4]),
            "low": float(parts[5]),
            "bid": float(parts[6]),
            "ask": float(parts[7]),
            "volume": int(parts[8]),           # 成交量
            "amount": float(parts[9]),         # 成交额
            "time": parts[30] + " " + parts[31],  # 日期 + 时间
        }
    except:
        return None


def realtime_loop(symbols, interval=1):
    """
    实时轮询行情（默认 1 秒刷新一次）
    """
    last_price = {}

    while True:
        text = fetch_sina_quote(symbols)
        print('text->',text)
        if not text:
            time.sleep(interval)
            continue
        
        lines = text.strip().split("\n")
        
        for sym, line in zip(symbols, lines):
            data = parse_sina_single(line)
            if not data:
                continue
            
            price = data["price"]
            now = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 判断是否更新
            if last_price.get(sym) != price:
                print(f"[{now}] {sym} {data['name']} 价格更新: {price}")
                last_price[sym] = price

        time.sleep(interval)


def download_15m(symbol):
    raw = fetch_sina_quote(symbol,15)
 
    if not raw:
        return    
    return convert_to_df(raw=raw)

def download_5m(symbol):
    raw = fetch_sina_quote(symbol,5)
 
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