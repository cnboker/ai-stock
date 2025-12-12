import os
# 必须最前面！！
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 防止 huggingface tokenizer 报 warning

import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time
from dash import Dash, dcc, html, Input, Output, callback
from chronos import ChronosPipeline
from quote import download_5m, download_15m
import pickle

# ========================== 全局配置 ==========================
PREDICTION_LENGTH = 10
UPDATE_INTERVAL_SEC = 60 * 15
CACHE_60D = {"5": {}, "15": {}}
PREDICTION_HISTORY = {}
HISTORY_FILE = "prediction_history.pkl"


pipeline = ChronosPipeline.from_pretrained(
    "./chronos-t5-large",           # 你本地路径
    device_map="auto",
    dtype=torch.float16,          # 必须 fp16，省 2GB+
    low_cpu_mem_usage=True,
)

# 优化 CUDA 行为
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

@torch.inference_mode()
def fast_chronos_predict_with_hs300(df: pd.DataFrame, period: str,ticker:str):
    """多变量预测（带沪深300），显存严格控制在 7.3GB 以内"""

    history_len = min(1024, len(df))  # Chronos 推荐 512~1024

    recent_df = df.iloc[-history_len:].copy()
    recent_df.index = recent_df.index.tz_localize(None)

    hs300_df = download_5m("sh000300") if period == "5" else download_15m("sh000300")
    hs300_df.index = pd.to_datetime(hs300_df.index).tz_localize(None)

    if hs300_df is not None:
        # 关键！用时间索引对齐，再插值填充缺失
        hs300_aligned = (
            hs300_df["close"]
            .reindex(recent_df.index, method="nearest")  # 先按时间对齐
            .interpolate(method="linear")                 # 线性插值补空
            #.bfill()           # 改这里
            .ffill()
            .values
        )    
    else:
        hs300_aligned = recent_df["close"].values  # 兜底

    # 3. 构造完美时间索引（Chronos 必须）
    freq_str = "5min" if period == "5" else "15min"

    perfect_index = pd.date_range(
        start=recent_df.index[0],
        periods=history_len,
        freq=freq_str
    )
    df_input = pd.DataFrame({
        "item_id": [ticker] * history_len,           # 必须是 list，不能是字符串, 
        "timestamp": perfect_index,
        "target": recent_df["close"].values,
        "volume": recent_df["volume"].values.astype(float),
        "hs300": hs300_aligned,
    })
    print("df_input 前10行：")
    print(df_input.head(10))
    print("\n")
    print("df_input 最后100行：")
    #这里显示的是假日期这是 Chronos 要求的“完美等间距时间序列”**，不是真实交易时间！
    print(df_input.tail(100))
    print(f"\ndf_input 总长度: {len(df_input)}, hs300 是否有 NaN: {df_input['hs300'].isna().sum()}")
    prediction = pipeline.predict_df(
        df_input,
        prediction_length=PREDICTION_LENGTH,
        num_samples=200,
        temperature=1.0,
        top_k=50,
        quantile_levels=[0.05, 0.5, 0.95],
    )

    
    low    = prediction["0.05"].values
    median = prediction["0.5"].values
    high   = prediction["0.95"].values
   
    
    del prediction, df_input
    torch.cuda.empty_cache()
    
    return low, median, high
# ========================== 工具函数 ==========================
def fetch_kline_df(ticker: str, period: str) -> pd.DataFrame:
    df = download_5m(ticker) if period == "5" else download_15m(ticker)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()

def get_60d_df(ticker: str, period: str) -> pd.DataFrame:
    key = f"{ticker}_{period}"
    if key not in CACHE_60D[period]:
        CACHE_60D[period][key] = fetch_kline_df(ticker, period)
    return CACHE_60D[period][key]

def append_prediction(ticker: str, future_df: pd.DataFrame):
    if ticker not in PREDICTION_HISTORY:
        PREDICTION_HISTORY[ticker] = future_df.copy()
    else:
        df = PREDICTION_HISTORY[ticker]
        df = pd.concat([df, future_df]).drop_duplicates(subset="datetime", keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)
        PREDICTION_HISTORY[ticker] = df
    # 自动持久化
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(PREDICTION_HISTORY, f)
    except:
        pass
    return PREDICTION_HISTORY[ticker]

# ========================== Dash App ==========================
app = Dash(__name__, title="Chronos 实时预测")

app.layout = html.Div([
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
            style={"width": "200px"}
        ),
        dcc.Dropdown(
            id="time-select",
            options=[{"label": "5分钟", "value": "5"}, {"label": "15分钟", "value": "15"}],
            value="5",
            style={"width": "120px"}
        ),
    ], style={"display": "flex", "justifyContent": "center", "gap": "20px", "padding": "20px"}),

    html.Div(id="last-update", style={"textAlign": "center", "color": "#0f0", "fontSize": "18px"}),

    dcc.Graph(id="live-graph", style={"height": "88vh"}),

    dcc.Interval(id="interval", interval=UPDATE_INTERVAL_SEC*1000, n_intervals=0),
], style={"backgroundColor": "#1e1e1e", "fontFamily": "Arial"})

# ========================== 主回调（极致优化） ==========================
@callback(
    Output("live-graph", "figure"),
    Output("last-update", "children"),
    Input("interval", "n_intervals"),
    Input("ticker-select", "value"),
    Input("time-select", "value"),
)
def update_graph(n_intervals, ticker, period):
    df = get_60d_df(ticker, period)
    if len(df) < 200:
        return go.Figure().update_layout(template="plotly_dark", title="数据不足"), "加载中..."

    # ==================== 预测（1.5秒） ====================
    close_vals = df["close"].values
    low, median, high = fast_chronos_predict_with_hs300(df,period,ticker)

    # ==================== 未来时间轴 ====================
    freq = "5min" if period == "5" else "15min"
    last_dt = df.index[-1].tz_localize(None)
    # 去时区
    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(minutes=int(period)),
        periods=PREDICTION_LENGTH,
        freq=freq
    )

    new_pred = pd.DataFrame({
        "datetime": future_index,
        "low": low, "median": median, "high": high
    })
    history_pred = append_prediction(ticker, new_pred)

    # ==================== 画图 ====================
    fig = go.Figure()

    # 历史预测区间
    if len(history_pred) > 5:
        fig.add_trace(go.Scatter(
            x=pd.concat([history_pred["datetime"], history_pred["datetime"][::-1]]),
            y=pd.concat([history_pred["low"], history_pred["high"][::-1]]),
            fill='toself',
            fillcolor='rgba(100,149,237,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(x=history_pred["datetime"], y=history_pred["median"],
                                mode='lines', line=dict(color="#FFA500", width=2, dash='dot'),
                                name='历史预测中位数'))

    # 今日实盘
    today = df[df.index.date == datetime.now().date()]
    if len(today) > 0:
        fig.add_trace(go.Scatter(x=today.index, y=today["close"],
                                mode='lines', line=dict(color="#00FF41", width=3),
                                name=f"{ticker} 实时"))

    # 最新预测边界
    fig.add_trace(go.Scatter(x=future_index, y=low, line=dict(color="gray", width=1), name="预测下界"))
    fig.add_trace(go.Scatter(x=future_index, y=high, line=dict(color="gray", width=1), name="预测上界"))
    fig.add_trace(go.Scatter(x=future_index, y=median, line=dict(color="#00BFFF", width=3), name="预测中位数"))

    # 坐标轴美化
    last_price = df["close"].iloc[-1]
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(range=[today.index[0] if len(today)>0 else df.index[0], future_index[-1]+pd.Timedelta(minutes=30)],
                   tickformat="%H:%M"),
        yaxis=dict(range=[last_price*0.88, last_price*1.12], title="价格 (元)"),
        hovermode="x unified",
        height=850,
        margin=dict(l=40,r=40,t=60,b=40),
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#1e1e1e"
    )

    update_time = datetime.now().strftime('%H:%M:%S')
    return fig, f"更新: {update_time} | {ticker} | Chronos-Large 实时预测"

# ========================== 启动（生产必关 debug） ==========================
if __name__ == "__main__":
    # 恢复历史预测记录
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                PREDICTION_HISTORY = pickle.load(f)
            print("历史预测记录已恢复")
        except:
            pass

    app.run(debug=True, port=8050)