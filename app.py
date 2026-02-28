import os
import threading

import pandas as pd
from global_state import equity_engine
from infra.core.context import TradingSession
from infra.core.runtime import RunMode
from plot.trade_monitor import live_stock_table
from position.live_position_loader import (
    live_positions_hot_load,
)
from predict.prediction_store import load_history
from trade.processor import execute_stock_analysis
from plot.draw import (
    draw_current_prediction,
    draw_prediction_band,
    draw_realtime_price,
    update_xaxes,
    update_yaxes,
)
from plot.annotation import generate_tail_label
from equity.equity_features import equity_features

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dash import Dash, dcc, html, Input, Output, callback, no_update

# ========================== 项目内模块 ==========================
from config.settings import TICKER_PERIOD, UPDATE_INTERVAL_SEC
from predict.time_utils import is_market_break
from data.loader import load_index_df
from plot.base import build_update_text, create_base_figure, finalize_figure

# app.py
from global_state import state_lock
from position.live_position_loader import live_positions_hot_load
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder

# ========================== Dash App ==========================
app = Dash(__name__, title="Chronos 实时预测")
# ====== 全局状态（只初始化一次） ======
position_mgr = create_position_manager(0, RunMode.LIVE)
eq_recorder = create_equity_recorder(RunMode.LIVE)

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
def update_graph(_):
    if is_market_break():
        return no_update, no_update

    period = TICKER_PERIOD
    hs300_df = load_index_df(str(period))
    
    prediction_tails = []

    with state_lock:
        positions = list(position_mgr.positions.items())
    fig = create_base_figure(len(positions))
    dfs = {}
    eq_feat = equity_features(eq_recorder.to_series())
    eq_decision = equity_engine.decide(eq_feat, position_mgr.has_any_position())
    equity_engine.log_equity_decision(eq_feat, eq_decision)
    session = TradingSession(
        run_mode=RunMode.LIVE,
        period=str(period),
        hs300_df=hs300_df,
        eq_feat=eq_feat,
        tradeIntent=eq_decision,
        position_mgr=position_mgr,
        eq_recorder=eq_recorder,
    )
    for index, (ticker, p) in enumerate(positions):
        try:
            result = execute_stock_analysis(ticker, session)
            
            decision = result["decision"]
            
            dfs[ticker] = {
                **decision,
                "low": result["low"][-1],
                "median": result["median"][-1],
                "high": result["high"][-1],
            }
           
            draw(result=result, fig=fig, index=index)

            tail = generate_tail_label(
                result["future_index"],
                result["median"],
                result["high"],
                index,
                result["name"],
            )
            if tail:
                prediction_tails.append(tail)

        except Exception as e:
            print(f"[WARN] {ticker} failed: {e}")

    # ===== 更新动态表格 =====
    df = pd.DataFrame(list(dfs.values()))
    #live_stock_table(df)

    # ===== 更新 equity 记录 =====
    eq_recorder.add(position_mgr.equity)

    # ===== 绘图最终处理 =====
    finalize_figure(fig, prediction_tails)
    return fig, build_update_text()


def draw(result, fig, index):
    draw_prediction_band(fig, result["history_pred"], index, result["name"])
    draw_realtime_price(fig, result["df"], index, result["name"])
    draw_current_prediction(
        fig,
        result["future_index"],
        result["low"],
        result["median"],
        result["high"],
        index,
        result["name"],
    )

    update_yaxes(fig, result["last_price"], index)
    update_xaxes(fig)


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
        args=(
            position_mgr,
            stop_event,
        ),
        daemon=True,
    )
    hotload_thread.start()

    app.run(debug=True, use_reloader=False, port=8050, host="localhost")
