import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
import pandas as pd
import time
from config.settings import ticker_name_map
'''
🟦 基础行情
ticker 代码
name 名称
last_price 最新价
day_low / day_high 日低 / 日高
ATR / range 波动
-------------------------------------------------
🟨 模型判断
model_mid / low / high 模型预测
raw_score 原始分数
regime 市场状态
-------------------------------------------------
🟥 风控 / Gate
gate / cooldown gate 值
Cooldown 是否冷却
Weak 弱信号过滤
Force  强制减仓
-------------------------------------------------
🟥 交易执行
action BUY / SELL / HOLD
confidence 置信度
strength 仓位强度
force_reduce 强制减仓
'''

# -------------------------------
# 阈值，可按需调整
# -------------------------------
THRESHOLDS = {
    "atr": 0,  # ATR > 0 高亮
    "model_score": 0.5,  # 模型分数大于 0.5 高亮
    "predicted_up": True,  # True 高亮
    "confidence": 0.6,  # 信心度大于 0.6 高亮
    "raw_score": 0.0,  # raw_score > 0 高亮
    "atr_is_zero": False,  # False 高亮
}

console = Console()
last_positions = {}  # 全局保存上一次持仓
live = None  # 全局 Live 对象
_stock_table = None


def init_stock_table() -> Table:
    global _stock_table
    if _stock_table is not None:
        return _stock_table

    table = Table(
        title="[bold green]⚡ 股票关键参数追踪 ⚡[/bold green]",
        show_lines=True,
        border_style="green",
    )

    columns = [
        "Ticker",
        "Name",
        "Entry",
        "Stop",
        "Take",
        "Pos",
        "ATR",
        "Low",
        "Mid",
        "High",
        "PredUp",
        "Regime",
        "Force",
        "Gate",
        "Raw",
        "Score",
        "Action",
        "Conf",
        "OK",
    ]

    for col in columns:
        table.add_column(col, justify="center", no_wrap=True)

    _stock_table = table
    return table


def make_stock_table(df: pd.DataFrame, last_positions: dict) -> Table:
    table = init_stock_table()
    table.rows.clear()

    for _, row in df.iterrows():
        ticker = row["ticker"]

        table.add_row(
            str(ticker),
            str(row.get("name", ticker_name_map.get(ticker, ticker))),
            fmt_price(row.get("entry_price")),
            fmt_price(row.get("stop_loss")),
            fmt_price(row.get("take_profit")),
            fmt(row.get("position_size"), 0),
            fmt(row.get("atr")),
            fmt_price(row.get("low")),
            fmt_price(row.get("median")),
            fmt_price(row.get("high")),
            fmt(row.get("predicted_up"),n=4),
            fmt_regime(row.get("regime")),
            fmt_bool(row.get("force_reduce")),
            fmt(row.get("gate_mult")),
            fmt(row.get("raw_score")),
            fmt(row.get("model_score")),
            fmt_action(row.get("action")),
            fmt(row.get("confidence")),
            fmt_bool(row.get("confirmed")),
        )

    return table

def fmt_price(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.2f}"

def fmt(x, n=2):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{n}f}"


def fmt_bool(x):
    if x is None:
        return "-"
    return "[yellow]True[/yellow]" if x else "[dim]False[/dim]"


def fmt_action(action):
    return {
        "BUY": "[bold green]BUY[/bold green]",
        "SELL": "[bold red]SELL[/bold red]",
        "HOLD": "[dim]HOLD[/dim]",
    }.get(action, action)


def fmt_regime(regime):
    return {
        "good": "[green]good[/green]",
        "neutral": "[yellow]neutral[/yellow]",
        "bad": "[red]bad[/red]",
    }.get(regime, regime)


# -------------------------------
# 主函数：实时刷新表格
# -------------------------------
def live_stock_table(df):
    global live, last_positions
    table = make_stock_table(df, last_positions)
    if live is None:
        # 第一次创建 Live 对象
        live = Live(table, console=console, refresh_per_second=1)
        live.start()
    else:
        # 后续只更新表格
        live.update(table)

    # 更新上次持仓
    last_positions = {row["ticker"]: row["position_size"] for _, row in df.iterrows()}
