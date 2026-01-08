from rich.console import Console
from rich.table import Table
from rich.live import Live
import pandas as pd
import time
from config.settings import ticker_name_map

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



# -------------------------------
# 构造表格函数
# -------------------------------
def make_stock_table(df: pd.DataFrame, last_positions: dict) -> Table:
  
    table = Table(
        title="[bold green]⚡ 股票关键参数追踪 ⚡[/bold green]",
        show_lines=True,
        border_style="green",
    )

    columns = [
        "Ticker",
        "Name",
        "Entry Price",
        "Stop Loss",
        "Take profit",
        "Position",
        "ATR",
        "Low Last",
        "Median Last",
        "High Last",
        "Predicted Up",
        "Regime",
        "Force Reduce",
        "Gate Mult",
        "Raw Score",
        "Model Score",
        "Action",
        "Confidence",
        "Confirmed",
    ]
    for col in columns:
        table.add_column(f"[bold green]{col}[/bold green]", justify="center")
    
    # 添加行
    for _, row in df.iterrows():
        ticker = row["ticker"]
        # ...这里和你原来的格式化逻辑一样...
        table.add_row(
            str(ticker),
            str(row.get("name", ticker_name_map.get(ticker, ticker))),
            fmt_price(row["entry_price"]),
            fmt_price(row["stop_loss"]),
            fmt_price(row["take_profit"]),
            str(row["position_size"]),
            str(row["atr"]),
            fmt_price(row["low"]),
            fmt_price(row["median"]),
            fmt_price(row["high"]),
            str(row["predicted_up"]),
            str(row["regime"]),
            str(row["force_reduce"]),
            str(row["gate_mult"]),
            str(row["raw_score"]),
            str(row["model_score"]),
            str(row["action"]),
            str(row["confidence"]),
            str(row["confirmed"]),
        )
    return table


def fmt_price(x):
    if x is None:
        return ""
    return f"{float(x):.2f}"


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


# -------------------------------
# 示例用法
# -------------------------------
# if __name__ == "__main__":
#     # 构造示例数据
#     df = pd.DataFrame([
#         {'ticker':'AAPL','name':'Apple','position':100,'atr':1.2,'atr_is_zero':False,
#          'model_score':0.6,'low_last':150,'median_last':152,'high_last':155,
#          'predicted_up':True,'regime':'good','gate_mult':1.0,'raw_score':0.05,
#          'action':'LONG','confidence':0.7},
#         {'ticker':'TSLA','name':'Tesla','position':50,'atr':0,'atr_is_zero':True,
#          'model_score':0.4,'low_last':250,'median_last':255,'high_last':260,
#          'predicted_up':False,'regime':'bad','gate_mult':0.5,'raw_score':-0.02,
#          'action':'HOLD','confidence':0.5}
#     ])

#     # 定义获取最新 df 的函数（替换成实盘/模拟盘逻辑）
#     def get_latest_df():
#         # 这里可以直接返回最新 df 或每次更新 df 后返回
#         # 示例：简单模拟持仓变化
#         df.at[0, 'position'] += 5  # 模拟 AAPL 持仓增加
#         df.at[0, 'action'] = 'LONG'
#         df.at[1, 'position'] -= 5  # 模拟 TSLA 持仓减少
#         df.at[1, 'action'] = 'SHORT'
#         return df

#     # 启动动态刷新
#     live_stock_table(get_latest_df, refresh_sec=1)
