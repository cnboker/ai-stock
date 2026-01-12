import numpy as np
import pandas as pd
from model_torch import is_torch_available,optional_inference_mode
from config.settings import MODEL_NAME
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
    prediction_length: int = 10,
    eq_feat: pd.DataFrame | None = None,
):
    """
    Chronos 多变量预测封装
    返回：low, median, high (np.ndarray)
    """

    if df is None or df.empty:
        raise ValueError("行情 df 为空")

    history_len = min(1024, len(df))
    recent_df = df.iloc[-history_len:].copy()
    recent_df.index = recent_df.index.tz_localize(None)

    # ========== HS300 对齐 ==========
    if hs300_df is not None and not hs300_df.empty:
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
    freq = "5min" if period == "5" else "15min"
    perfect_index = pd.date_range(
        start=recent_df.index[0],
        periods=history_len,
        freq=freq,
    )

    # Equity 特征对齐时间轴
    eq_feat_aligned = eq_feat.reindex(perfect_index).dropna()
   
    if eq_feat_aligned.empty:
        
        df_input = pd.DataFrame(
            {
                "item_id": [ticker] * history_len,
                "timestamp": perfect_index,
                "target": recent_df["close"].values.astype(float),
                # ===== 原有协变量 =====
                "volume": recent_df["volume"].values.astype(float),
                "hs300": hs300_series.astype(float),            
            }
        )
    else:
        df_input = pd.DataFrame(
            {
                "item_id": [ticker] * history_len,
                "timestamp": perfect_index,
                "target": recent_df["close"].values.astype(float),
                # ===== 原有协变量 =====
                "volume": recent_df["volume"].values.astype(float),
                "hs300": hs300_series.astype(float),
                # ===== Equity 协变量 =====
                "eq_ret": eq_feat_aligned["eq_ret"].values,
                "eq_drawdown": eq_feat_aligned["eq_drawdown"].values,
                "eq_slope": eq_feat_aligned["eq_slope"].values,
                # ===== 权重增强（重复通道）=====
                # 重复列 = 多通道 = 更高 attention
                "eq_ret_r1": eq_feat_aligned["eq_ret"].values,
                "eq_ret_z_r1": eq_feat_aligned["eq_ret_z"].values,
            }
        )

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
    # print("is_t5_style", is_t5_style)
    # print(low, median, high)
    # 显存清理（只在 CUDA）
    del pred
    if is_torch_available():
        import torch
        torch.cuda.is_available()
        torch.cuda.empty_cache()

    latest_price = df["close"].iloc[-1]
    model_score = model_score_from_quantiles(
        low=low,
        median=median,
        high=high,
        latest_price=latest_price,
    )

    atr = calc_atr(df)
    #print('model_score', model_score)
    return PredictionResult(
        low = low,
        median = median,
        high = high,
        model_score = model_score,
        atr=atr
    )
    #return low, median, high, model_score


def model_score_from_quantiles(low: np.ndarray, median: np.ndarray, high: np.ndarray, latest_price: float) -> np.ndarray:
    """
    返回 [0, 1] 的 model_score，支持 array
    """
    low = np.asarray(low)
    median = np.asarray(median)
    high = np.asarray(high)

    width = (high - low) / np.maximum(np.abs(median), 1e-6)
    deviation = np.abs(median - latest_price) / np.maximum(np.abs(median), 1e-6)
    score = deviation / (width + 1e-6)

    score_float = float(np.clip(np.mean(score), 0.0, 1.0))
    return score_float