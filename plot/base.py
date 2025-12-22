from datetime import time
import pandas as pd
from plotly.subplots import make_subplots
from plot.annotation import create_annotation
from plot.draw import subplot_position


def create_base_figure():
    """
    创建 2×2 基础子图结构
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.03,
    )

    fig.update_layout(
        margin=dict(l=50, r=40, t=20, b=30),
        height=1100,
    )

    fig.update_yaxes(title_standoff=4)
    return fig


def finalize_figure(
    fig,
    prediction_tails,
):

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
        gridcolor="#030303",
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
        annotations=create_annotation(prediction_tails),
        hovermode="closest",  # 你已经改成 closest，很好
        spikedistance=1000,
        hoverdistance=1000,
        height=1200,        
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#1e1e1e",
        dragmode="pan",
        #  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
