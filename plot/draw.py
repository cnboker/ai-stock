import plotly.graph_objects as go
import pandas as pd
from config.settings import COLORS
from datetime import datetime
import numpy as np


def subplot_position(index: int):
    row = 1 if index < 2 else 2
    col = 1 if index % 2 == 0 else 2
    return row, col


# history draw
def draw_prediction_band(fig, history_pred, index, name):
    if len(history_pred) < 5:
        return

    row, col = subplot_position(index)
    color = COLORS[index]

    fig.add_trace(
        go.Scatter(
            x=pd.concat([history_pred["datetime"], history_pred["datetime"][::-1]]),
            y=pd.concat([history_pred["low"], history_pred["high"][::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.2),
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
            line=dict(color=color, width=2, dash="dot"),
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
            customdata=np.stack((history_pred["low"], history_pred["high"]), axis=1),
        ),
        row=row,
        col=col,
    )


def hex_to_rgba(hex_color, alpha=0.25):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def draw_realtime_price(fig, df, index, name):
    today_df = df[df.index.date == datetime.now().date()]
    row, col = subplot_position(index)

    fig.add_trace(
        go.Scatter(
            x=today_df.index,
            y=today_df["close"],
            mode="lines",
            line=dict(color=COLORS[index], width=3),
            name=f"{name} 实时",
            legendgroup=name,
        ),
        row=row,
        col=col,
    )


def draw_current_prediction(fig, future_index, low, median, high, index, name):
    row, col = subplot_position(index)
    color = COLORS[index]

    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=median,
            mode="lines",
            line=dict(color=color, width=4),
            name=f"{name} 预测",
            legendgroup=name,
            customdata=np.stack((low, high), axis=1),
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
            fillcolor=hex_to_rgba(color, 0.25),
            line=dict(width=0),
            legendgroup=name,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def update_yaxes(fig, latest_price, index):
    row, col = subplot_position(index)
    y_padding = 0.05
    y_min = latest_price * (1 - y_padding)
    y_max = latest_price * (1 + y_padding)
    fig.update_yaxes(
       # title_text="相对价格 (%)" if col == 1 else "",  # 只在左侧显示标题
        range=[y_min, y_max],  # 每个子图独立范围（可自定义）
        tickformat=".1f",
        side="left",
        showgrid=True,
        gridcolor="#333",
        zeroline=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(title_standoff=2)

#跨天加日期   
def update_xaxes(fig):
    fig.update_xaxes(
        type="date",
        tickformatstops=[
            dict(dtickrange=[None, 1000 * 60 * 60 * 6], value="%H:%M"),
            dict(dtickrange=[1000 * 60 * 60 * 6, 1000 * 60 * 60 * 24], value="%m-%d %H:%M"),
            dict(dtickrange=[1000 * 60 * 60 * 24, None], value="%Y-%m-%d"),
        ]
    )