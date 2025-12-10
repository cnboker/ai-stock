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

PREDICTION_LENGTH = 5         # 预测未来几根 15m K 线
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


#使用 yfinance 获取 15m K 线数据，period="60d" 保证了有足够的历史数据作为 Chronos 的 context
def fetch_15m_df(ticker: str):
    """
    使用 yfinance 获取 15 分钟 K 线数据，返回 DataFrame（index 为 DatetimeIndex）。
    自动处理 yfinance 返回可能的多层列名（MultiIndex）。
    """
 
    global CACHE_60D

    df = download_15m(ticker)
    if df is None or df.empty:
        return pd.DataFrame()
    # 转为 DatetimeIndex（保险）
    df.index = pd.to_datetime(df.index)
    #print(df.head())
    return df

def get_60d_df(ticker, period):
    if ticker not in CACHE_60D[period]:
        CACHE_60D[period][ticker] = fetch_kline_df(ticker, period)
    return CACHE_60D[period][ticker]

# def get_simulated_context(df_context, df_now_day, n_intervals):
#     # n_intervals 从 0 开始，所以取前 n_intervals+1 条
#     df_live = df_now_day.iloc[: n_intervals + 1]
#     df_combined = pd.concat([df_context, df_live])
#     return df_combined
 
#处理了 yfinance 返回的列名兼容性问题（如 MultiIndex），鲁棒性强。
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
    if time(11,30) <= now.time() <= time(13,0):
        fig = go.Figure().update_layout(template="plotly_dark", title="午盘休息时间，不绘图")
        return fig, f"当前时间 {now.strftime('%H:%M:%S')}，午盘休息中"

    df = get_60d_df(ticker, time_select)

    #df.index = df.index.tz_convert('Asia/Shanghai')
    last_day = df.index[-1].date()
    df_now_day = df[df.index.date == last_day]    # 当天全部15m数据

    if df.empty:
        fig = go.Figure().update_layout(template="plotly_dark", title="无数据")
        return fig, "无数据"
    fig = go.Figure().update_layout(template="plotly_dark", title="实时盘面")
    close_series = extract_close_series(df)
    print('close_series',close_series)

    # ==================== Chronos 预测 ====================
    context = torch.tensor(close_series.values, dtype=torch.float32)

    forecast = pipeline.predict(
        inputs=context,
        prediction_length=PREDICTION_LENGTH,
        num_samples=1000
    )

    samples = forecast[0].cpu().numpy()                                      # (1000, 10)
    #计算 1000 个样本在每个时间点上的第 5%、50%（中位数）、95% 分位数，定义 90% 置信区间
    low, median, high = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)     # 90% 置信区间

    # 强制转 list + float（防止任何标量）
    def L(x): 
        return np.asarray(x, dtype=float).flatten().tolist()

    low, median, high = L(low), L(median), L(high)

    # 未来时间轴（关键！必须是真实交易时间）获取历史数据的最后一个时间点。
    last_dt = close_series.index[-1]
    #生成未来 10 根 K 线的正确时间戳，这是 Plotly 正确绘图的关键。
    freq = "5min" if time_select == "5" else "15min"
    future_index = pd.date_range(last_dt, periods=PREDICTION_LENGTH, freq=freq)
    print('future_index->', future_index)
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

   # 置信区间 CI

   
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
        title=f"Chronos 15分钟实时预测（{PREDICTION_LENGTH}根未来K线）",
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
