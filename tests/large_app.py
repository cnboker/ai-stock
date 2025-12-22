import os

from signal import make_signal
from utils import get_next_trading_times

# 必须最前面！！
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止 huggingface tokenizer 报 warning

import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time
from dash import Dash, dcc, html, Input, Output, callback
from chronos import ChronosPipeline
from quote import download_5m, download_15m
import pickle
from plotly.subplots import make_subplots
from dash import no_update
from predict.chronos_predict import chronos_predict

# ========================== 全局配置 ==========================
PREDICTION_LENGTH = 10
UPDATE_INTERVAL_SEC = 60 * 5
CACHE_60D = {"5": {}, "15": {}}
PREDICTION_HISTORY = {}
HISTORY_FILE = "prediction_history.pkl"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
pipeline = ChronosPipeline.from_pretrained(
    "./chronos-t5-large",  # 你本地路径
    device_map="auto",
    dtype=torch.float16,  # 必须 fp16，省 2GB+
    low_cpu_mem_usage=True,
)

# 优化 CUDA 行为
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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


def append_prediction(ticker: str, new_pred: pd.DataFrame):
    new_pred = new_pred.copy()
    new_pred["ticker"] = ticker  # 加上标记方便调试

    if ticker not in PREDICTION_HISTORY:
        PREDICTION_HISTORY[ticker] = new_pred
    else:
        existing = PREDICTION_HISTORY[ticker]
        # 只保留不与新预测重叠的部分 + 新预测
        last_existing_time = existing["datetime"].iloc[-1]
        new_start_time = new_pred["datetime"].iloc[0]

        if new_start_time <= last_existing_time:
            # 有重叠：裁剪旧数据，只保留到重叠前
            existing = existing[existing["datetime"] < new_start_time]

        updated = pd.concat([existing, new_pred]).reset_index(drop=True)
        PREDICTION_HISTORY[ticker] = updated

    # 持久化（加 try 更稳健）
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(PREDICTION_HISTORY, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print("持久化失败:", e)

    return PREDICTION_HISTORY[ticker]


# ========================== Dash App ==========================
app = Dash(__name__, title="Chronos 实时预测")

app.layout = html.Div(
    [
        html.Div(
            id="last-update",
            style={"textAlign": "center", "color": "#0f0", "fontSize": "18px"},
        ),
        dcc.Graph(id="live-graph", style={"height": "88vh"}),
        dcc.Interval(id="interval", interval=UPDATE_INTERVAL_SEC * 1000, n_intervals=0),
    ],
    style={"backgroundColor": "#1e1e1e", "fontFamily": "Arial"},
)


# ========================== 主回调（极致优化） ==========================
@callback(
    Output("live-graph", "figure"),
    Output("last-update", "children"),
    Input("interval", "n_intervals"),
    # Input("ticker-select", "value"),
    # Input("time-select", "value"),
)
def update_graph(n_intervals):

    now = datetime.now()
    current_time = now.time()
    if time(11, 30) <= current_time < time(13, 0):
        return no_update

    period = "5"
    ALL_TICKERS = [
        {"code": "sh600446", "name": "金证股份"},
        {"code": "sz300142", "name": "沃生生物"},
        {"code": "sh600771", "name": "广誉远"},
        {"code": "sz002137", "name": "实益达"},
    ]
    hs300_df = download_5m("sh000300") if period == "5" else download_15m("sh000300")
    hs300_df.index = pd.to_datetime(hs300_df.index).tz_localize(None)
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05)
    # 先收集每只股票的最新价格和时间（用于贴标签）
    prediction_tails = []
    colors = ["#00ff00", "#ff8800", "#00cccc", "#ff66cc"]

    for i, stock in enumerate(ALL_TICKERS):
        ticker = stock["code"]
        name = stock["name"]
        row = 1 if i < 2 else 2
        col = 1 if i % 2 == 0 else 2
        color = ["#00ff00", "#ff8800", "#00cccc", "#ff66cc"][i]
        try:
            df = fetch_kline_df(ticker, period)
            low, median, high = chronos_predict(
                df=df,
                hs300_df=hs300_df,
                ticker=ticker,
                period=period,
                prediction_length=PREDICTION_LENGTH,
            )

            # 真实最新价格（收盘价）
            latest_price = float(df["close"].iloc[-1])
            latest_time = df.index[-1]
            print("latest_time", latest_time)

            # ==================== 未来时间轴 ====================
            freq = "5min" if period == "5" else "15min"
            last_dt = df.index[-1].tz_localize(None)
            # 去时区
            # future_index = get_next_trading_times(last_dt, PREDICTION_LENGTH, freq)
            # 解决午休,周末等问题
            future_index = get_next_trading_times(
                start_dt=last_dt,
                n=PREDICTION_LENGTH,
                freq="5min",
            )

            # future_index = pd.date_range(
            #     start=last_dt + pd.Timedelta(minutes=int(period)),
            #     periods=PREDICTION_LENGTH,
            #     freq=freq,
            #     tz=None,
            # )

            print("future_index", future_index)

            new_pred = pd.DataFrame(
                {"datetime": future_index, "low": low, "median": median, "high": high}
            )
            history_pred = append_prediction(ticker, new_pred)
            # 必须是严格单调递增的（即按时间从早到晚排序）

            print("history_pred", history_pred)
            # 历史预测区间
            if len(history_pred) > 5:
                # 保险：确保时间有序
                history_pred = history_pred.sort_values("datetime").reset_index(
                    drop=True
                )

                # 动态填充颜色（可选）
                base_color = colors[i]
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                fillcolor = hex_to_rgba(color, 0.2)

                # 填充区间（95% 置信带）
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat(
                            [history_pred["datetime"], history_pred["datetime"][::-1]]
                        ),
                        y=pd.concat([history_pred["low"], history_pred["high"][::-1]]),
                        fill="toself",
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=name,
                    ),
                    row=row,
                    col=col,
                )

                # 中位数线（带 hover）
                fig.add_trace(
                    go.Scatter(
                        x=history_pred["datetime"],
                        y=history_pred["median"],
                        mode="lines",
                        line=dict(color=colors[i], width=2, dash="dot"),
                        name="历史预测中位数",
                        legendgroup=name,
                        opacity=0.6,
                        hovertemplate=(
                            f"<b>{name} 历史预测</b><br>"
                            "时间: %{x|%H:%M}<br>"
                            "预测中位数: <b>%{y:.2f}</b> 元<br>"
                            "95% 区间: [%{customdata[0]:.2f}, %{customdata[1]:.2f}] 元"
                            "<extra></extra>"
                        ),
                        customdata=np.stack(
                            (history_pred["low"], history_pred["high"]), axis=1
                        ),
                    ),
                    row=row,
                    col=col,
                )
            # 只画三条线：今日实盘 + 本次预测中位数 + 本次置信区间
            today_df = df[df.index.date == datetime.now().date()]

            # 记录用于后面打标签
            prediction_tails.append(
                {
                    "x": future_index[-1],  # 预测线的最后一个时间点
                    "y": high[-1],  # 预测中位数的最后一个值
                    "name": name,
                    "price": median[-1],
                    "color": colors[i],
                }
            )

            fig.add_trace(
                go.Scatter(
                    name=f"{name} 实时",
                    x=today_df.index,
                    y=today_df["close"],
                    mode="lines",
                    line=dict(color=colors[i], width=3),
                    legendgroup=name,
                    hoverinfo="none",
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=future_index,
                    y=median,
                    mode="lines",
                    line=dict(color=colors[i], width=4, dash="solid"),
                    name=f"{name} 预测",
                    legendgroup=name,
                    customdata=np.stack((low, high), axis=1),  # shape = (10, 2)
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "时间: %{x|%H:%M}<br>"
                        "预测中位数: <b>%{y:.2f}</b> 元<br>"
                        "95% 下轨: <b>%{customdata[0]:.2f}</b> 元<br>"
                        "95% 上轨: <b>%{customdata[1]:.2f}</b> 元"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(future_index) + list(future_index)[::-1],
                    y=list(high) + list(low)[::-1],
                    fill="toself",
                    fillcolor=hex_to_rgba(colors[i], 0.25),
                    line=dict(width=0),
                    legendgroup=name,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # 更新Y坐标
            # Y轴：动态 ±12%（比你原来的 ±12% 更宽，防止截断）
            y_padding = 0.05
            y_min = latest_price * (1 - y_padding)
            y_max = latest_price * (1 + y_padding)
            fig.update_yaxes(
                title_text="相对价格 (%)" if col == 1 else "",  # 只在左侧显示标题
                range=[y_min, y_max],  # 每个子图独立范围（可自定义）
                tickformat=".1f",
                side="left",
                showgrid=True,
                gridcolor="#333",
                zeroline=False,
                row=row,
                col=col,
            )

        except Exception as e:
            print(f"{ticker} 预测失败: {e}")
            continue

    annotations = create_annotation(prediction_tails)

    # ==================== 坐标轴范围（关键修复！）===================
    today_date = pd.Timestamp.now().date()
    # X轴：固定显示当天 9:30 - 15:00（带时区）
    x_start = pd.Timestamp.combine(today_date, time(9, 30)).tz_localize("Asia/Shanghai")
    x_end = pd.Timestamp.combine(today_date, time(15, 0)).tz_localize("Asia/Shanghai")
    # 实现x轴滚动效果
    fig.update_xaxes(
        range=[x_start, x_end],
        autorange=False,
        fixedrange=False,
        constrain="domain",
        showgrid=True,
        gridcolor="#333333",
        title="",
        tickformat="%H:%M",
        showspikes=True,
        spikecolor="#555",
        spikesnap="cursor",
        spikemode="across",
        # rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="当天", step="day", stepmode="backward"),
                    dict(count=5, label="5天", step="day", stepmode="backward"),
                    dict(count=10, label="10天", step="day", stepmode="backward"),
                    # dict(step="all", label="全部"),  # 如不想显示全部，可注释掉
                ]
            ),
            bgcolor="#1e1e1e",
            activecolor="#333333",
            font=dict(color="#ffffff"),
            bordercolor="#444444",
            borderwidth=1,
        ),
    )
    # 在 update_xaxes 之后，加这几行强制同步
    for i in range(1, 5):  # 对应 xaxis, xaxis2, xaxis3, xaxis4
        fig.update_xaxes(
            range=[x_start, x_end],
            autorange=False,
            row=(1 if i <= 2 else 2),
            col=(1 if i % 2 == 1 else 2) if i <= 4 else 1,
        )
        # 或者直接：
        fig.layout[f'xaxis{i if i > 1 else ""}'].range = [x_start, x_end]
        fig.layout[f'xaxis{i if i > 1 else ""}'].autorange = False
    fig.update_layout(
        template="plotly_dark",
        annotations=annotations,
        hovermode="closest",  # 你已经改成 closest，很好
        spikedistance=1000,
        hoverdistance=1000,
        height=1200,
        margin=dict(l=60, r=100, t=20, b=30),
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#1e1e1e",
        dragmode="pan",
        #  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    update_time = datetime.now().strftime("%H:%M:%S")
    return fig, f"更新: {update_time} | {ticker} | Chronos-Large 实时预测"


def create_annotation(prediction_tails):
    # ==================== 2×2 子图专用尾部标签（完美不乱） ====================
    annotations = []

    # 子图坐标映射
    subplot_refs = [
        ("x1", "y1"),  # 第1只：左上
        ("x2", "y2"),  # 第2只：右上
        ("x3", "y3"),  # 第3只：左下
        ("x4", "y4"),  # 第4只：右下
    ]

    for i, tail in enumerate(prediction_tails):
        name = tail["name"]
        price = tail["price"]
        color = tail["color"]
        x = tail["x"]
        y = tail["y"]

        xref, yref = subplot_refs[i]  # 每只股票用自己的子图坐标系

        annotations.append(
            dict(
                x=x,
                y=y,
                xref=xref,
                yref=yref,
                text=f" → <b>{name} {price:.2f}</b>",
                font=dict(size=15, color="white", family="Consolas"),
                arrowhead=2,
                arrowcolor=color,
                arrowwidth=3,
                ax=-80,  # 箭头指向左，指向预测终点
                ay=0,
                opacity=0.96,
                xanchor="left",
                yanchor="middle",
            )
        )
    return annotations


def hex_to_rgba(hex_color, alpha=0.25):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# 直接把这整个块粘到你的代码最后（在 if __name__ == "__main__ 之前）

app.clientside_callback(
    """
    function(fig) {
        // 每次 figure 更新都重新绑定事件
        setTimeout(() => {
            const plot = document.querySelector('.js-plotly-plot');
            if (!plot) return;

            // 清除旧事件（防止重复绑定）
            plot.onplotly_hover = null;
            plot.onplotly_unhover = null;

            plot.on('plotly_hover', (data) => {
                if (!data.points?.[0]) return;
                const group = data.points[0].data.legendgroup;
                document.querySelectorAll('.scatterlayer .trace').forEach((t,i) => {
                    const traceGroup = plot.data[i]?.legendgroup;
                    t.style.opacity = (traceGroup === group) ? '1' : '0.15';
                });
            });

            plot.on('plotly_unhover', (data) => {
                if (!data.points?.[0]) return;
                const group = data.points[0].data.legendgroup;
                document.querySelectorAll('.scatterlayer .trace').forEach((t, i) => {
                    const traceGroup = plot.data[i]?.legendgroup;
                    t.style.opacity = (traceGroup === group) ? '1' : '0.15';
                });
            });
        }, 100);

        return window.dash_clientside.no_update;
    }
    """,
    Output("live-graph", "id"),
    Input("live-graph", "figure"),
)
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
