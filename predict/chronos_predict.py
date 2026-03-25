import numpy as np
import pandas as pd
from model_torch import optional_inference_mode
from config.settings import MODEL_NAME,LOOKBACK_WINDOW
from predict.chronos_model import load_chronos_model
from predict.predict_result import PredictionResult
from predict.price_alpha import chronos2_to_large_style
from predict.time_utils import calc_atr

@optional_inference_mode()
def run_prediction(
    df: pd.DataFrame,
    hs300_df: pd.DataFrame | None,
    ticker: str = "",
    period: str = "5",
    prediction_length: int = 5,
    eq_feat: pd.DataFrame | None = None,
):
    """
    Chronos 多变量预测封装
    返回：low, median, high (np.ndarray)
    """

    if df is None or df.empty:
        raise ValueError("行情 df 为空")
 
    history_len = min(LOOKBACK_WINDOW, len(df))
    recent_df = df.iloc[-history_len:].copy()
    recent_df.index = recent_df.index.tz_localize(None)

    # ========== HS300 对齐 ==========
    if hs300_df is not None and not hs300_df.empty:
        hs300_df = hs300_df.tail(history_len)
        hs300_series = (
            hs300_df["close"]
            .reindex(recent_df.index, method="nearest")
            .interpolate("linear")
            .ffill()
            .values
        )
    else:
        hs300_series = recent_df["close"].values

    # ========== Chronos 要求的“完美时间索引” ==========
    freq = f"{period}min"
    perfect_index = pd.date_range(
        start=recent_df.index[0],
        periods=history_len,
        freq=freq,
    )

    # 修改代码如下：
    if eq_feat is not None:
        # 只有当 eq_feat 存在时才进行重采样和对齐
        perfect_index = pd.date_range(
            start=recent_df.index[0],
            periods=history_len,
            freq=freq,
        )
        eq_feat_aligned = eq_feat.reindex(perfect_index).dropna()
    else:
        # 如果是 None（比如回测刚开始），给一个空的 Series 或 DataFrame
        # 确保后面的程序不会因为找不到变量而崩溃
        eq_feat_aligned = pd.Series(dtype='float64') 
        print("DEBUG: eq_feat 为空，跳过特征对齐")

   # 1. 价格取对数收益率或标准化，让模型对波动更敏感
    # 2. HS300 转化成相对于价格的超额变动
    hs300_relative = (hs300_series / hs300_series[0]) / (recent_df["close"].values / recent_df["close"].values[0])
    
    # 3. 成交量标准化（Z-Score 或 移动平均比）
    vol_mean = recent_df["volume"].mean()
    vol_std = recent_df["volume"].std() + 1e-6
    vol_norm = (recent_df["volume"].values - vol_mean) / vol_std

    if eq_feat_aligned.empty:
        df_input = pd.DataFrame({
            "item_id": [ticker] * history_len,
            "timestamp": perfect_index,
            "target": recent_df["close"].values.astype(float),
            "volume": vol_norm.astype(float), # 传标准化后的量
            "hs300": hs300_relative.astype(float), # 传相对强弱
        })
    else:
        # Equity 特征也需要 Z-Score，否则模型不知道 0.01 的收益算不算多
        df_input = pd.DataFrame({
            "item_id": [ticker] * history_len,
            "timestamp": perfect_index,
            "target": recent_df["close"].values.astype(float),
            "volume": vol_norm.astype(float),
            "hs300": hs300_relative.astype(float),
            "eq_drawdown": eq_feat_aligned["eq_drawdown"].values * 10, # 放大回撤权重
            "eq_slope": eq_feat_aligned["eq_slope"].values,
        })

    pipeline = load_chronos_model(MODEL_NAME)
    # ========== Chronos 推理 ==========
    pred = pipeline.predict_df(
        df=df_input,
        prediction_length=prediction_length,
    )
    #print('pre->',pred)
    q10 = pred["0.1"].values
    q50 = pred["0.5"].values
    q90 = pred["0.9"].values
    low, median, high = q10, q50, q90

    is_t5_style = MODEL_NAME.startswith("chronos-t5")

    if not is_t5_style:
        context = recent_df["close"].values
        assert len(context) >= len(q50), "context too short"

        low, median, high = chronos2_to_large_style(
            q10=q10,
            q50=q50,
            q90=q90,
            context=context,
        )

    latest_price = df["close"].iloc[-1]
    atr = calc_atr(df)
    model_score = model_score_from_quantiles_trend(
        low=low,
        median=median,
        high=high,
        latest_price=latest_price,
        atr=atr
    )    
    
    return PredictionResult(
        low = low,
        median = median,
        high = high,
        model_score = model_score,
        atr=atr,
        price = latest_price
    )
    #return low, median, high, model_score

def model_score_from_quantiles_trend(low, median, high, latest_price, atr):
    """
    强化版评分：引入 ATR 归一化和置信区间偏离度
    """
    low = np.asarray(low)
    median = np.asarray(median)
    high = np.asarray(high)
    atr = max(atr, 1e-6) # 防止除零

    # 1. 预测趋势强度 (用预测终点相对于起点的涨幅，除以 ATR 归一化)
    # 这样无论是 3 块的 ETF 还是 300 块的股票，分数值域是对齐的
    pred_move = (median[-1] - median[0]) / atr
    
    # 2. 价格位置评分 (看现价在预测区间 low-high 中的位置)
    # 如果现价靠近 low，说明下方空间小，风险大；靠近 high 说明上方空间大
    range_width = (high[-1] - low[-1]) + 1e-6
    position_score = (high[-1] - latest_price) / range_width
    
    # 3. 现价与中位数的偏离度 (Mean Reversion 因子)
    # 如果现价远低于中位数，说明有向上修复动力
    deviation = (median[0] - latest_price) / atr

    # --- 组合评分权重 (可由 Optuna 进一步微调) ---
    # 趋势(0.5) + 位置(0.3) + 偏离(0.2)
    raw_score = (pred_move * 0.5) + (position_score * 0.3) + (deviation * 0.2)

    # 4. 映射到 0~1 区间，并加入逻辑“死区”
    # 如果 raw_score 太小（波动不足），强制回归 0.5 (即 HOLD)
    if abs(raw_score) < 0.05: 
        return 0.5
    
    score_float = float(np.clip((raw_score + 1) / 2, 0.0, 1.0))
    return score_float

