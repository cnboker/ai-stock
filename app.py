import os
import time
from datetime import datetime,time

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from chronos import ChronosPipeline
from sina import download_15m,download_5m

PREDICTION_LENGTH = 10         # 预测未来几根 15m K 线
UPDATE_INTERVAL_SEC = 60  * 15     # 刷新间隔（秒），可改小（注意 API 限制）
CACHE_60D = {
    "5": {},
    "15": {}
}
            # 多股票缓存
PREDICTION_HISTORY = {}     # 多股票预测历史
# 设备与 dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ---------------------- 加载模型（只加载一次） ----------------------
print("Loading Chronos pipeline (this may take a while)...")
# 使用国内镜像（强烈推荐）,指定 HF 的全局缓存位置（强制写到本地）
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_DATASETS_CACHE"] = "./hf_cache"


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=DEVICE,
    torch_dtype=DTYPE,
    cache_dir="./hf_cache",
    local_files_only=False,   # 第一次联网下载
)

# ---------------------- Dash App ----------------------
app = dash.Dash(__name__, title="Chronos 实时预测")

app.layout = html.Div(
    [
        html.H2("多股票 Chronos 实时预测", style={"textAlign": "center","color":"white"}),

        # ---- 单独的 flex 容器，只包含两个 Dropdown ----
        html.Div(
            [
                dcc.Dropdown(
                    id="ticker-select",
                    options=[
                        {"label": "电工合金", "value": "sz300697"},
                        {"label": "沃生生物", "value": "sz300142"},
                        {"label": "广誉远", "value": "sh600771"},
                        {"label": "实益达", "value": "sz002137"},
                         {"label": "db", "value": "sh600693"},
                    ],
                    value="sz300697",
                    style={"width": "180px"}
                ),

                dcc.Dropdown(
                    id="time-select",
                    options=[
                        {"label": "5m", "value": "5"},
                        {"label": "15m", "value": "15"}
                    ],
                    value="5",
                    style={"width": "120px"}
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "gap": "12px",
                "marginBottom": "15px",
            }
        ),

        html.Div(id="last-update", style={"textAlign": "center"}),

        dcc.Graph(id="live-graph"),

        dcc.Interval(id="interval-component", interval=15*60*1000, n_intervals=0),
    ]
)

# ---------------------- 辅助函数 ----------------------
def to_list(x):
    """确保任何输入都转成 list"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, pd.Series):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if np.isscalar(x):  # float64, int64, python float, etc.
        return [x]
    # 其他可迭代类型
    try:
        return list(x)
    except:
        return [x]


def get_60d_df(ticker, period):
    if ticker not in CACHE_60D[period]:
        CACHE_60D[period][ticker] = fetch_kline_df(ticker, period)
    return CACHE_60D[period][ticker]

# def get_simulated_context(df_context, df_now_day, n_intervals):
#     # n_intervals 从 0 开始，所以取前 n_intervals+1 条
#     df_live = df_now_day.iloc[: n_intervals + 1]
#     df_combined = pd.concat([df_context, df_live])
#     return df_combined
 
def extract_close_series(df):
    # Chronos forecast 中值
    if "median" in df.columns:
        return df["median"]
    # 如果你未来还会混入真实 close，可以保留兼容性
    if "close" in df.columns:
        return df["close"]
    raise KeyError(f"df has no median/close columns: {df.columns}")

def fetch_kline_df(ticker: str, period: str):
    if period == "5":
        df = download_5m(ticker)
    else:
        df = download_15m(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    return df


@app.callback(
    Output("live-graph", "figure"),
    Output("last-update", "children"),
    Input("interval-component", "n_intervals"),
    Input("ticker-select", "value"),     # 多股票：加这个
    Input("time-select", "value"),
)

def update_graph(n_intervals, ticker, time_select):
    print(f"第 {n_intervals} 次刷新 @ {datetime.now().strftime('%H:%M:%S')}")
    
    now = pd.Timestamp.now(tz="Asia/Shanghai").to_pydatetime()  # 获取当前北京时间
    fig = go.Figure()
    if time(11,30) <= now.time() <= time(13,0):
        fig.update_layout(template="plotly_dark", title="午盘休息时间，不绘图")
        return fig, f"当前时间 {now.strftime('%H:%M:%S')}，午盘休息中"

    df = get_60d_df(ticker, time_select)

    #df.index = df.index.tz_convert('Asia/Shanghai')
    last_day = df.index[-1].date()
    df_now_day = df[df.index.date == last_day]    # 当天全部15m数据

    if df.empty:
        fig.update_layout(template="plotly_dark", title="无数据")
        return fig, "无数据"
    
    fig.update_layout(template="plotly_dark", title="实时盘面")

    close_series = extract_close_series(df)
        # ==================== Chronos 预测（替换段） ====================
    # Basic checks
    series = close_series.astype(float)
    print("series length:", len(series))
    print(series.describe())
    if len(series) < 64:
        fig.update_layout(template="plotly_dark", title="序列太短，无法预测")
        return fig, f"数据点太少（{len(series)}），需要更多历史数据"

    std = float(np.std(series.values))
    if std == 0 or np.isclose(std, 0.0):
        fig.update_layout(template="plotly_dark", title="序列方差为0，无法预测")
        return fig, f"序列方差为 0 ，无法预测（常数序列）"

    print("first 20 values of series:", series.values[:20])

    # ---- 建模 log-return（更稳定） ----
    logp = np.log(series.values)
    rets = np.diff(logp)          # length = N-1
    last_price = float(series.values[-1])

    # pipeline.predict 要求 inputs 是 batch list
    # 可以直接传 list of floats，pipeline 会处理 dtype/device
    # Chronos 要求 batch 输入，因此必须是 list[tensor]
    context = torch.tensor(rets, dtype=torch.float32)
    inputs = [context]   # ★★★ 关键修复
    # ---------- IMPORTANT: ensure pipeline device choice ----------
    # If you created pipeline with device="cpu" above, this is safe.
    # If pipeline lives on GPU (not recommended for small model in Dash), ensure inputs are tensors on same device:
    #    inputs = [torch.tensor(rets, dtype=torch.float32, device=pipeline.device)]
    #
    # Here we use list of floats which is simplest and device-safe.
    forecast = pipeline.predict(
        inputs=inputs,
        prediction_length=PREDICTION_LENGTH,
        num_samples=1000
    )

    # 统一为 numpy array
    samples = np.asarray(forecast)
    print("forecast raw shape:", samples.shape)

    # 处理不同返回维度情况 -> 期望最后得到 (num_samples, pred_len)
    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]   # (num_samples, pred_len)
        print("adjusted samples shape ->", samples.shape)
    elif samples.ndim == 2:
        pass
    elif samples.ndim == 1 and samples.size == PREDICTION_LENGTH:
        samples = np.expand_dims(samples, axis=0)
        print("single sample returned, shape ->", samples.shape)
    else:
        print("WARNING: unrecognized forecast shape:", samples.shape)

    # samples now = predicted log-returns (num_samples, pred_len)
    # 把每条 sample 还原为 price 路径： cum_log -> exp -> * last_price
    price_paths = []
    for s in samples:
        cum_log = np.cumsum(s)
        future_price = last_price * np.exp(cum_log)
        price_paths.append(future_price)
    price_paths = np.asarray(price_paths)  # (num_samples, pred_len)

    # 在 price 空间计算 quantiles
    low, median, high = np.quantile(price_paths, [0.05, 0.5, 0.95], axis=0)

    # 转成 list 便于绘图
    def to_float_list(x):
        return np.asarray(x, dtype=float).flatten().tolist()

    low = to_float_list(low)
    median = to_float_list(median)
    high = to_float_list(high)

    # 生成未来时间轴（从 last_dt + freq 开始）
    last_dt = close_series.index[-1]
    freq = "5min" if time_select == "5" else "15min"
    start_dt = last_dt + pd.Timedelta(freq)
    future_index = pd.date_range(start=start_dt, periods=PREDICTION_LENGTH, freq=freq)

    print("example first sample (log-returns):", samples[0])
    print("example first sample (prices):", price_paths[0])
    print("median prices:", median)
    print("low[0], high[0]:", low[0], high[0])

    # ----------------- 添加到历史 -----------------
    new_forecast_df = pd.DataFrame({
        "datetime": future_index,
        "median": median,
        "low": low,
        "high": high
    })

     # 3. 历史预测置信区间这样可以保证 CI 区域封闭，不会出现空缺
    df_ci =  append_prediction(ticker,new_forecast_df=new_forecast_df)
    
    x = list(df_ci["datetime"]) + list(df_ci["datetime"])[::-1]
    y = list(df_ci["low"]) + list(df_ci["high"])[::-1]

    fig.add_trace(go.Scatter(
        x=x,
        y=df_ci["low"],
        mode="lines",
        line=dict(color="gray", width=1),
        name="预测下界"
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=df_ci["high"],
        mode="lines",
        line=dict(color="gray", width=1),
        name="预测上界"
    ))

    #今日曲线始终绘制 df_last_day
    fig.add_trace(go.Scatter(
        x=df_now_day.index,
        y=to_list(df_now_day["close"]),
        mode="lines",
        name=f"{ticker} 今日价格",
        line=dict(width=2)
    ))


    # 历史预测中位数（也强制 list）
    fig.add_trace(go.Scatter(
        x=df_ci["datetime"],
        y=to_list(df_ci["median"]),
        mode="lines+markers",
        name="历史预测中位数",
        line=dict(color="#FF6F00", width=2),
        marker=dict(size=6)
    ))

  
    # ==================== 布局（强制显示所有数据）===================
    from datetime import  timedelta

    # 假设 last_day 是 datetime.date 类型，例如 df.index[-1].date()
    last_day = pd.Timestamp.now().date()  # 示例当天日期

    x_start = pd.Timestamp.combine(last_day, time(9, 30)).tz_localize('Asia/Shanghai')
    x_end   = pd.Timestamp.combine(last_day, time(15, 0)).tz_localize('Asia/Shanghai')
    latest_price = float(close_series.iloc[-1])
    fig.update_layout(
        template="plotly_dark",
        title=f"Chronos实时预测（{PREDICTION_LENGTH}根未来K线）",
        xaxis=dict(
            range=[x_start, x_end],  # 固定显示 9:30 ~ 15:00
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)"
        ),
       yaxis=dict(
            range=[
                latest_price * 0.90,   # -10%
                latest_price * 1.10    # +10%
            ],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=800
    )


    # 强制显示所有点（最关键的一行！治愈空白图神器）
    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikecolor="white")

    return fig, f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def append_prediction(ticker, new_forecast_df):
    global PREDICTION_HISTORY

    # 如果没有就创建
    if ticker not in PREDICTION_HISTORY:
        PREDICTION_HISTORY[ticker] = new_forecast_df
    else:
        # 合并并去重
        df = PREDICTION_HISTORY[ticker]
        df = pd.concat([df, new_forecast_df])
        df = df.drop_duplicates(subset="datetime").sort_values("datetime")
        PREDICTION_HISTORY[ticker] = df

    return PREDICTION_HISTORY[ticker]

# ---------------------- 启动 ----------------------
if __name__ == "__main__":
    # 注意：在生产环境请把 debug=False
    app.run(debug=True, port=8050)
