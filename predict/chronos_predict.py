import pandas as pd
import torch

from config.settings import MODEL_NAME
from predict.chronos_model import load_chronos_model
from predict.price_alpha import chronos2_to_large_style


@torch.inference_mode()
def run_prediction(
    df: pd.DataFrame,
    hs300_df: pd.DataFrame | None,
    ticker: str,
    period: str = "5",
    prediction_length: int = 10,
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

    df_input = pd.DataFrame(
        {
            "item_id": [ticker] * history_len,
            "timestamp": perfect_index,
            "target": recent_df["close"].values.astype(float),
            "volume": recent_df["volume"].values.astype(float),
            "hs300": hs300_series.astype(float),
        }
    )
    pipeline = load_chronos_model(MODEL_NAME)
    # ========== Chronos 推理 ==========
    pred = pipeline.predict_df(
        df=df_input,
        prediction_length=prediction_length,
    )

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

    # 显存清理（只在 CUDA）
    del pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return low, median, high
