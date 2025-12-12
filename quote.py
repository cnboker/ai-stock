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