# ========================== 必须最前面（CUDA / Torch 配置） ==========================
import asyncio
import os
import threading
import traceback

from position.LivePositionLoader import live_positions_hot_load
from position.position_manager2 import position_mgr
from predict.prediction_store import load_history

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================== 基础库 ==========================
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, callback, no_update

# ========================== 项目内模块 ==========================
from config.settings import TICKER_PERIOD, UPDATE_INTERVAL_SEC, ALL_TICKERS
from predict.time_utils import is_market_break
from data.loader import load_index_df
from plot.base import create_base_figure, finalize_figure
from update_graph import process_single_stock, build_update_text

# ========================== Dash App ==========================
app = Dash(__name__, title="Chronos 实时预测")

app.layout = html.Div(
    [
        html.Div(
            id="last-update",
            style={
                "textAlign": "center",
                "color": "#00ff99",
                "fontSize": "18px",
                "padding": "6px",
            },
        ),
        dcc.Graph(
            id="live-graph",
            style={"height": "88vh"},
            config={
                "displayModeBar": True,
                "scrollZoom": True,
            },
        ),
        dcc.Interval(
            id="interval",
            interval=UPDATE_INTERVAL_SEC * 1000,
            n_intervals=0,
        ),
    ],
    style={
        "backgroundColor": "#1e1e1e",
        "fontFamily": "Arial",
    },
)

# ========================== 主回调（极薄） ==========================
@app.callback(
    Output("live-graph", "figure"),
    Output("last-update", "children"),
    Input("interval", "n_intervals"),
)
def update_graph(n_intervals):
    """
    Dash 回调入口：
    - 只负责调度
    - 不关心任何细节
    """

    # 午休不更新（避免空预测 & 闪图）
    # if is_market_break():
    #     return no_update, no_update

    period = TICKER_PERIOD

    # 加载指数（一次）
    hs300_df = load_index_df(period)

    # 创建空 Figure
    fig = create_base_figure()

    prediction_tails = []

    # === 核心循环：每只股票 ===
    for index, (ticker,p) in enumerate(position_mgr.positions.copy().items()):
        try:
            tail = process_single_stock(
                fig=fig,
                ticker=ticker,
                index=index,
                period=period,
                hs300_df=hs300_df,
            )
            if tail:
                prediction_tails.append(tail)

        except Exception as e:
            print(f"[WARN] {ticker} 处理失败: {e}")
            #traceback.print_stack()
    # 统一收尾（annotation / layout）
   # print('prediction_tails',prediction_tails)
    finalize_figure(fig, prediction_tails)

    return fig, build_update_text()

# ========================== 客户端 hover 联动（保持你原来的高级体验） ==========================
app.clientside_callback(
    """
    function(fig) {
        setTimeout(() => {
            const plot = document.querySelector('.js-plotly-plot');
            if (!plot) return;

            plot.onplotly_hover = null;
            plot.onplotly_unhover = null;

            plot.on('plotly_hover', (data) => {
                if (!data.points?.[0]) return;
                const group = data.points[0].data.legendgroup;
                document.querySelectorAll('.scatterlayer .trace').forEach((t,i) => {
                    const g = plot.data[i]?.legendgroup;
                    t.style.opacity = (g === group) ? '1' : '0.15';
                });
            });

            plot.on('plotly_unhover', () => {
                document.querySelectorAll('.scatterlayer .trace')
                    .forEach(t => t.style.opacity = '1');
            });
        }, 80);

        return window.dash_clientside.no_update;
    }
    """,
    Output("live-graph", "id"),
    Input("live-graph", "figure"),
)

# ========================== 启动 ==========================
if __name__ == "__main__":
    print("🚀 Chronos Dash 启动中...")
    load_history()
    stop_event = threading.Event()

    hotload_thread = threading.Thread(
        target=live_positions_hot_load,
        args=(),
        daemon=True
    )
    hotload_thread.start()
    app.run(debug=True, port=8050, host="0.0.0.0")
