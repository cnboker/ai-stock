import numpy as np
import pandas as pd

def zscore(s: pd.Series, win: int):
    return (s - s.rolling(win).mean()) / (s.rolling(win).std() + 1e-8)

def equity_features(eq)-> pd.DataFrame:
    # 基础检查：如果数据长度不足以支撑最大的窗口 (50)，直接返回空表
    # 这能避免 rolling 产生全 NaN 导致的后续逻辑崩溃
  # 1. 安全检查逻辑
    if eq is None:
        return pd.DataFrame(columns=["eq_ret", "eq_ret_z", "eq_drawdown", "eq_slope"])
    
    # 2. 确保是 Series 且长度足够
    # 使用 .shape[0] 或 len() 都是安全的
    if len(eq) < 50:
        return pd.DataFrame(columns=["eq_ret", "eq_ret_z", "eq_drawdown", "eq_slope"])

    # 3. 排除全 NaN 的情况
    if eq.isna().all():
        return pd.DataFrame(columns=["eq_ret", "eq_ret_z", "eq_drawdown", "eq_slope"])
    # 1. 计算收益率
    ret = eq.pct_change(fill_method=None)
    
    # 2. 计算回撤
    drawdown = eq / eq.cummax() - 1
    
    # 3. 滚动 Z-Score (窗口 50)
    # 此时 len(eq) >= 50，所以这里至少会有一个有效值
    rolling_50 = ret.rolling(50)
    eq_ret_z = (ret - rolling_50.mean()) / rolling_50.std().replace(0, np.nan)
    
    # 4. 滚动斜率 (窗口 20)
    # 使用 raw=True 避开 Series 对象的歧义判断
    def calc_slope(y):
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    eq_slope = eq.rolling(20).apply(calc_slope, raw=True)
   
    df = pd.DataFrame({
        "eq_ret": ret,
        "eq_ret_z": eq_ret_z,
        "eq_drawdown": drawdown,
        "eq_slope": eq_slope
    })
    #print('get_metrics', df)
    # dropna 会删掉前 49 行（因为 Z-Score 窗口是 50）
    # 如果总长度只有 10，这里结果就是空，需要确保下游能处理空 DataFrame
    return df.dropna()