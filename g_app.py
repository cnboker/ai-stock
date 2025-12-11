import os
import time
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from chronos import ChronosPipeline

from sina import download_15m, download_5m

# 你的 sina 下载函数（假设已定义好
# from sina import download_15m, download_5m

PREDICTION_LENGTH = 10
UPDATE_INTERVAL_SEC = 60 * 15
CACHE_60D = {"5": {}, "15": {}}
PREDICTION_HISTORY = {}  # {ticker: pd.DataFrame()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ---------------------- 加载模型 ----------------------
print("Loading Chronos model...")
os.environ["HF_HOME"] = "./hf_cache"

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=DEVICE,
    dtype=DTYPE,
    cache_dir="./hf_cache",
)
print(pipeline.__class__.__name__)
# ---------------------- Dash App ----------------------
app = dash.Dash(__name__, title="Chronos 实时预测")

app.layout = html.Div([
    html.H2("多股票 Chronos 实时预测", style={"textAlign": "center", "color": "white", "background": "#1e1e1e", "padding": "20px"}),

    html.Div([
        dcc.Dropdown(
            id="ticker-select",
            options=[
                {"label": "电工合金", "value": "sz300697"},
                {"label": "沃生生物", "value": "sz300142"},
                {"label": "广誉远", "value": "sh600771"},
                {"label": "实益达", "value": "sz002137"},
                {"label": "东百集团", "value": "sh600693"},
            ],
            value="sz300697",
            style={"width": "200px", "color": "#000"}
        ),
        dcc.Dropdown(
            id="time-select",
            options=[{"label": "5分钟", "value": "5"}, {"label": "15分钟", "value": "15"}],
            value="5",
            style={"width": "120px", "color": "#000"}
        ),
    ], style={"display": "flex", "justifyContent": "center", "gap": "20px", "margin": "20px"}),

    html.Div(id="last-update", style={"textAlign": "center", "color": "#888", "marginBottom": "10px"}),

    dcc.Graph(id="live-graph", style={"height": "800px"}),

    dcc.Interval(id="interval-component", interval=UPDATE_INTERVAL_SEC*1000, n_intervals=0),
], style={"backgroundColor": "#1e1e1e"})

# ---------------------- 工具函数 ----------------------
def fetch_kline_df(ticker: str, period: str) -> pd.DataFrame:
    # 这里替换成你的实际下载函数
    if period == "5":
        df = download_5m(ticker)    # 你自己的函数
    else:
        df = download_15m(ticker)
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def get_60d_df(ticker, period):
    key = f"{ticker}_{period}"
    if key not in CACHE_60D[period]:
        CACHE_60D[period][key] = fetch_kline_df(ticker, period)
    return CACHE_60D[period][key]

def append_prediction(ticker: str, new_forecast_df: pd.DataFrame):
    """追加新的预测，保持历史预测连续"""
    if ticker not in PREDICTION_HISTORY:
        PREDICTION_HISTORY[ticker] = new_forecast_df.copy()
    else:
        df = PREDICTION_HISTORY[ticker]
        df = pd.concat([df, new_forecast_df], ignore_index=False)
        df = df.drop_duplicates(subset="datetime", keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)
        PREDICTION_HISTORY[ticker] = df
    return PREDICTION_HISTORY[ticker]

# ---------------------- 主回调 ----------------------
@app.callback(
    Output("live-graph", "figure"),
    Output("last-update", "children"),
    Input("interval-component", "n_intervals"),
    Input("ticker-select", "value"),
    Input("time-select", "value"),
)

def update_graph(n_intervals, ticker, time_select):
    now = datetime.now()
    now_time = now.time()

    # 非交易时间
    if not ((time(9,30) <= now_time <= time(11,30)) or (time(13,0) <= now_time <= time(15,0))):
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="非交易时间，暂停更新")
        return fig, f"非交易时间：{now.strftime('%H:%M:%S')}"

    df = get_60d_df(ticker, time_select)
    df = df.sort_index()
    if df.empty or len(df) < 100:
        fig = go.Figure().update_layout(template="plotly_dark", title="数据不足")
        return fig, "数据加载失败"

    # ==================== 沪深300缓存===================
    if not hasattr(update_graph, "hs300_cache"):
        update_graph.hs300_cache = {"5": None, "15": None}

    if update_graph.hs300_cache[time_select] is None:
        hs300_df = download_5m("sh000300") if time_select == "5" else download_15m("sh000300")
        if hs300_df is not None and not hs300_df.empty:
            hs300_df.index = pd.to_datetime(hs300_df.index)
            update_graph.hs300_cache[time_select] = hs300_df["close"]
        else:
            update_graph.hs300_cache[time_select] = df["close"]  # 兜底

    hs300_series = update_graph.hs300_cache[time_select].reindex(df.index, method="nearest")

    # ==================== Chronos预测（宇宙最稳版）===================
    # history_len = min(1800, len(df))
    freq_str = "5min" if time_select == "5" else "15min"

    history_len = min(1024, len(df))  # Chronos 推荐 512~1024

    recent_df = df.iloc[-history_len:].copy()
    recent_df.index = recent_df.index.tz_localize(None)

    recent_hs300 = hs300_series.iloc[-history_len:]
    recent_hs300.index = recent_hs300.index.tz_localize(None)
    last_ts = df.index[-1].tz_localize(None)
    perfect_index = pd.date_range(
        end=last_ts,
        periods=history_len,
        freq=freq_str        
    )
    print('perfect_index->',perfect_index)
    # 关键！不要传 timestamp！用默认整数索引代替
    df_input = pd.DataFrame({
        "item_id": ticker, 
        "timestamp": perfect_index,
        "target": recent_df["close"].values,
        "volume": recent_df["volume"].values.astype(float),
        "hs300": recent_hs300.values,
    })

   
    forecast = pipeline.predict_df(
        df_input,
        prediction_length=PREDICTION_LENGTH,
        num_samples=1000,
        temperature=1.3,
        quantile_levels=[0.05, 0.5, 0.95],
    )
    # 正确列名！
    low    = forecast["0.05"].values
    median = forecast["0.5"].values
    high   = forecast["0.95"].values
   

    # ==================== 未来时间轴（关键修复！）===================
    last_dt = df.index[-1].tz_localize(None)  # 去时区
    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(minutes=int(time_select)),
        periods=PREDICTION_LENGTH,
        freq=freq_str,
        tz=None
    )

    new_forecast = pd.DataFrame({
        "datetime": future_index,
        "low": low,
        "median": median,
        "high": high
    })

    # 追加历史（修复重复调用bug）
    df_history_pred = append_prediction(ticker, new_forecast)
    print('df_history_pred',df_history_pred)
    fig = go.Figure()

    # 历史预测置信区间（闭合填充）—— 强烈建议用这个更稳的写法
    if len(df_history_pred) > 5:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_history_pred["datetime"], df_history_pred["datetime"][::-1]]),
            y=pd.concat([df_history_pred["low"], df_history_pred["high"][::-1]]),
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.18)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        print("历史预测长度:", len(df_history_pred))
        print(df_history_pred.tail(10))
        # 历史中位数轨迹
        fig.add_trace(go.Scatter(
            x=df_history_pred["datetime"],
            y=df_history_pred["median"],
            mode='lines',
            line=dict(color="#FF8C00", width=2, dash='dot'),
            name='历史预测中位数'
        ))

    # 今日实盘曲线
    today_date = pd.Timestamp.now().date()
    
    df_today = df[df.index.date == today_date]
    if len(df_today) > 0:
        fig.add_trace(go.Scatter(
            x=df_today.index,
            y=df_today["close"],
            mode='lines',
            line=dict(color="#00FF41", width=3),
            name=f"{ticker} 实时价格"
        ))

    # 最新预测中位数（高亮）
    fig.add_trace(go.Scatter(
        x=future_index,
        y=median,
        mode='lines+markers',
        line=dict(color="#FFD700", width=5),
        marker=dict(size=8),
        name="最新预测中位数"
    ))

    # ==================== 坐标轴范围（关键修复！）===================
    last_price = df["close"].iloc[-1]

    # X轴：固定显示当天 9:30 - 15:00（带时区）
    x_start = pd.Timestamp.combine(today_date, time(9, 30)).tz_localize('Asia/Shanghai')
    x_end   = pd.Timestamp.combine(today_date, time(15, 0)).tz_localize('Asia/Shanghai')

    # Y轴：动态 ±12%（比你原来的 ±12% 更宽，防止截断）
    y_padding = 0.12
    y_min = last_price * (1 - y_padding)
    y_max = last_price * (1 + y_padding)

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Chronos 实时预测 · {ticker} · {time_select}分钟线 · 未来{PREDICTION_LENGTH}根K线",
            x=0.5,
            font=dict(size=20, color="#FFD700")
        ),
        xaxis=dict(
            range=[x_start, x_end],
            showgrid=True,
            gridcolor="#333333",
            title="",
            tickformat="%H:%M"
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#333333",
            title="价格 (元)"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=850,
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#1e1e1e"
    )

    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor")

    return fig, f"更新时间：{now.strftime('%H:%M:%S')} | {ticker} | Chronos-2 多变量预测"

# ---------------------- 启动 ----------------------
if __name__ == "__main__":
    # 注意：在生产环境请把 debug=False
    app.run(debug=True, port=8050)
