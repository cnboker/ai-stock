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
    "atr": 0,                  # ATR > 0 高亮
    "model_score": 0.5,        # 模型分数大于 0.5 高亮
    "predicted_up": True,      # True 高亮
    "confidence": 0.6,         # 信心度大于 0.6 高亮
    "raw_score": 0.0,          # raw_score > 0 高亮
    "atr_is_zero": False,      # False 高亮
}
console = Console()
last_positions = {}  # 全局保存上一次持仓
live = None          # 全局 Live 对象

# -------------------------------
# 构造表格函数
# -------------------------------
def make_stock_table(df: pd.DataFrame, last_positions: dict) -> Table:
    """
    Hack 风格股票表格
    df: 最新数据
    last_positions: {ticker: 上次持仓} 用于高亮持仓变化
    """
    table = Table(title="[bold green]⚡ 股票关键参数追踪 ⚡[/bold green]", show_lines=True, border_style="green")

    columns = ['Ticker','Name','Position','ATR',
               'Low Last','Median Last','High Last','Predicted Up',
               'Regime','Gate Mult','Raw Score','Action','Confidence','Confirmed']
    for col in columns:
        table.add_column(f"[bold green]{col}[/bold green]", justify="center")

    for _, row in df.iterrows():
        ticker = row['ticker']

        # -------------------------------
        # 内部高亮函数
        # -------------------------------
        def format_cell(col_name, value):
            if col_name not in THRESHOLDS:
                return str(value)
            thresh = THRESHOLDS[col_name]
            if isinstance(value, (int, float)):
                return f"[green]{value}[/green]" if value > thresh else str(value)
            elif isinstance(value, bool):
                return f"[green]{value}[/green]" if value == thresh else str(value)
            else:
                return str(value)

        # -------------------------------
        # 持仓变化高亮
        # -------------------------------
        pos_change = False
        if ticker in last_positions and row['position_size'] != last_positions[ticker]:
            pos_change = True
        position_display = f"[yellow]{row['position_size']}[/yellow]" if pos_change else str(row['position_size'])

        # -------------------------------
        # 动作高亮
        # -------------------------------
        action_color = {
            "LONG": "bright_green",
            "SHORT": "red",
            "REDUCE": "yellow",
            None: "white"
        }.get(row['action'], "white")
        action_display = f"[{action_color}]{row['action']}[/{action_color}]"

        table.add_row(
            format_cell('Ticker', ticker),
            format_cell('Name', ticker_name_map.get(ticker,ticker)),
            position_display,
            format_cell('atr', row['atr']),
            format_cell('low_last', fmt_price(row['low'])),
            format_cell('median_last', fmt_price(row['median'])),
            format_cell('high_last', fmt_price(row['high'])),
            format_cell('predicted_up', row['predicted_up']),
            format_cell('regime', row['regime']),
            format_cell('gate_mult', row['gate_mult']),
            format_cell('raw_score', row['raw_score']),
            action_display,
            format_cell('confidence', row['confidence']),
            format_cell('confirmed', row['confirmed']),
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
    last_positions = {row['ticker']: row['position_size'] for _, row in df.iterrows()}

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
