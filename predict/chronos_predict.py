import numpy as np
import pandas as pd
from model_torch import optional_inference_mode
from config.settings import MODEL_NAME, LOOKBACK_WINDOW
from predict.chronos_model import load_chronos_model
from predict.predict_result import PredictionResult
from predict.price_alpha import chronos2_to_large_style
from predict.time_utils import calc_atr


@optional_inference_mode()
def run_prediction(
    df: pd.DataFrame,
    hs300_df: np.ndarray | None,
    ticker: str = "",
    period: str = "30",
    prediction_length: int = 5,
    eq_feat: pd.DataFrame | None = None,
):
    """
    Chronos 多变量预测封装
    返回：low, median, high (np.ndarray)
    """
    # 预生成一个足够长的时间轴，比如 2000 个点，覆盖所有可能的历史窗口
    GLOBAL_PERFECT_INDEX = pd.date_range(
        "2026-01-01", periods=2000, freq=f"{period}min"
    )

    history_len = min(LOOKBACK_WINDOW, len(df))
    
    target_values = df["close"].values[-history_len:].astype(float)
    volume_values = df["volume"].values[-history_len:].astype(float)
    hs300_df = hs300_df[-history_len:].astype(float) if hs300_df is not None else None
    # recent_df = df.iloc[-history_len:].copy()
    # recent_df.index = recent_df.index.tz_localize(None)
    # ========== 极简标准化 (Vectorized) ==========
    vol_norm = (volume_values - np.mean(volume_values)) / (np.std(volume_values) + 1e-6)

    # 假设 hs300_series_aligned 已经在外部对齐好长度
    # 推荐写法：使用 .iloc[0] 获取第一行数据
    h_start = hs300_df[0] if hs300_df[0] != 0 else 1e-6
    t_start = target_values[0] if target_values[0] != 0 else 1e-6
    hs300_relative = (hs300_df / h_start) / (target_values / t_start)
    # ========== 构建输入 (跳过复杂的 pd.date_range) ==========
    # Chronos 其实对 timestamp 的要求只要是连续的即可
    # 如果不是为了展示，甚至可以用伪造的 range 提升速度
    perfect_index = GLOBAL_PERFECT_INDEX[:history_len]

    df_input = {
        "item_id": [ticker] * history_len,
        "timestamp": perfect_index.values,  # 取 values
        "target": target_values.astype(float),
        "volume": vol_norm.astype(float),
        "hs300": hs300_relative.astype(float),
    }
    
    if eq_feat is not None and not eq_feat.empty:
        # 1. 先提取 values
        drawdown_vals = eq_feat["eq_drawdown"].values * 10
        slope_vals = eq_feat["eq_slope"].values
        
        # 2. 关键修复：如果长度不足 history_len，在左侧补 0
        if len(drawdown_vals) < history_len:
            pad_width = history_len - len(drawdown_vals)
            drawdown_vals = np.pad(drawdown_vals, (pad_width, 0), 'constant', constant_values=0)
            slope_vals = np.pad(slope_vals, (pad_width, 0), 'constant', constant_values=0)
        
        # 3. 如果长度超过 history_len，取最后一段
        df_input["eq_drawdown"] = drawdown_vals[-history_len:]
        df_input["eq_slope"] = slope_vals[-history_len:]
    else:
        # 兜底：填充全 0 数组
        df_input["eq_drawdown"] = np.zeros(history_len)
        df_input["eq_slope"] = np.zeros(history_len)
    #print(f"🔍 输入数据构建完成，长度: {history_len} | 包含特征: {list(df_input.keys())}")
    input_df = pd.DataFrame(df_input)

    pipeline = load_chronos_model(MODEL_NAME)
    # ========== Chronos 推理 ==========
    pred = pipeline.predict_df(
        df=input_df,
        prediction_length=prediction_length,
    )
    # print('pre->',pred)
    q10 = pred["0.1"].values
    q50 = pred["0.5"].values
    q90 = pred["0.9"].values
    low, median, high = q10, q50, q90

    is_t5_style = MODEL_NAME.startswith("chronos-t5")

    if not is_t5_style:
        # context = recent_df["close"].values
        context = target_values
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
        low=low, median=median, high=high, latest_price=latest_price, atr=atr
    )

    return PredictionResult(
        low=low,
        median=median,
        high=high,
        model_score=model_score,
        atr=atr,
        price=latest_price,
    )
    # return low, median, high, model_score


def model_score_from_quantiles_trend(low, median, high, latest_price, atr):
    """
    强化版评分：引入 ATR 归一化和置信区间偏离度
    """
    low = np.asarray(low)
    median = np.asarray(median)
    high = np.asarray(high)
    atr = max(atr, 1e-6)  # 防止除零

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
