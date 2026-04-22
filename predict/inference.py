import numpy as np
import pandas as pd
from infra.core.runtime import GlobalState
from infra.utils.time_profile import timer_decorator
from model_torch import optional_inference_mode
from config.settings import MODEL_NAME
from predict.predict_result import PredictionResult
from predict.price_alpha import chronos2_to_large_style
from predict.time_utils import calc_atr
from predict.factory import PredictorFactory # 引入工厂

@optional_inference_mode()

def run_prediction(
    df: pd.DataFrame,
    hs300_df: np.ndarray | None,
    ticker: str = "",
    period: str = "30",
    prediction_length: int = 5,
    eq_feat: pd.DataFrame | None = None,
):
    # 1. 准备时间戳 (保留你的 GLOBAL_PERFECT_INDEX 逻辑)
    GLOBAL_PERFECT_INDEX = pd.date_range("2026-01-01", periods=2000, freq=f"{period}min")
    history_len = min(GlobalState.chronos_context_length, len(df))
    recent_df = df.iloc[-history_len:]
    perfect_index = GLOBAL_PERFECT_INDEX[:history_len]

    # 2. 构建特征输入 (保留你所有的原始特征工程)
    if MODEL_NAME.startswith("kronos"):
        df_input = {
            "item_id": [ticker] * history_len,
            "timestamp": perfect_index,
            "open": recent_df["open"].values.astype(float),
            "high": recent_df["high"].values.astype(float),
            "low": recent_df["low"].values.astype(float),
            "close": recent_df["close"].values.astype(float),
            "volume": recent_df["volume"].values.astype(float),
        }
        
    else:
        # --- 保留 Chronos 特有的 HS300/Equity 特征计算 ---
        target_values = df["close"].values[-history_len:].astype(float)
        volume_values = df["volume"].values[-history_len:].astype(float)
        hs300_data = hs300_df[-history_len:].astype(float) if hs300_df is not None else None
        
        vol_norm = (volume_values - np.mean(volume_values)) / (np.std(volume_values) + 1e-6)
        h_start = hs300_data[0] if hs300_data is not None and hs300_data[0] != 0 else 1e-6
        t_start = target_values[0] if target_values[0] != 0 else 1e-6
        hs300_relative = (hs300_data / h_start) / (target_values / t_start) if hs300_data is not None else np.ones(history_len)

        df_input = {
            "item_id": [ticker] * history_len,
            "timestamp": perfect_index,
            "target": target_values,
            "volume": vol_norm,
            "hs300": hs300_relative,
        }
        # ... 这里保留你关于 eq_feat 的补齐逻辑 ...
        if eq_feat is not None and not eq_feat.empty:
            drawdown_vals = eq_feat["eq_drawdown"].values * 10
            slope_vals = eq_feat["eq_slope"].values
            if len(drawdown_vals) < history_len:
                pad_width = history_len - len(drawdown_vals)
                drawdown_vals = np.pad(drawdown_vals, (pad_width, 0), 'constant')
                slope_vals = np.pad(slope_vals, (pad_width, 0), 'constant')
            df_input["eq_drawdown"], df_input["eq_slope"] = drawdown_vals[-history_len:], slope_vals[-history_len:]
        else:
            df_input["eq_drawdown"], df_input["eq_slope"] = np.zeros(history_len), np.zeros(history_len)

    input_df = pd.DataFrame(df_input)

    # 3. 推理阶段：切换为适配器调用
    adapter = PredictorFactory.get_adapter()
    pred = adapter.predict(input_df, prediction_length)

    # 4. 解析预测值 (统一输出结构)
    if MODEL_NAME.startswith("kronos"):
       
        # 核心进化：用真实的 std 替换死权重 (0.99/1.01)
        # 在正态分布下，0.1分位数 ≈ mean - 1.28 * std
        # 在正态分布下，0.9分位数 ≈ mean + 1.28 * std
        
        q50 = pred["median"].values
        current_std = pred["std"].values
        
        # 这里的 1.28 可以根据你对风险的容忍度调整：
        # 1.28 = 80% 置信区间 (更能捕捉波动)
        # 1.64 = 90% 置信区间 (更稳健)
        q10 = q50 - 1.28 * current_std
        q90 = q50 + 1.28 * current_std
        
        # 保留 log 用于调试，你会发现这比原来的 0.99 灵敏得多
        # print(f"Kronos 动态区间: low={q10[-1]:.2f}, mid={q50[-1]:.2f}, high={q90[-1]:.2f}, std_ratio={current_std[-1]/q50[-1]:.4f}")
    else:
        q10, q50, q90 = pred["0.1"].values, pred["0.5"].values, pred["0.9"].values

    # 5. T5 样式修正 (保留逻辑)
    low, median, high = q10, q50, q90
    if not MODEL_NAME.startswith("kronos") and not MODEL_NAME.startswith("chronos-t5"):
        low, median, high = chronos2_to_large_style(q10=q10, q50=q50, q90=q90, context=target_values)

    # 6. 评分逻辑 (保留你强化版的评分函数)
    latest_price = GlobalState.tickers_price[ticker]
    atr = calc_atr(df)
    model_score = model_score_from_quantiles_trend(low, median, high, latest_price, atr)

    return PredictionResult(low=low, median=median, high=high, model_score=model_score, atr=atr, std=pred["std"].values if MODEL_NAME.startswith("kronos") else None )


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
