from venv import logger
from position.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.signal_debouncer import debouncer_manager
from strategy.signal_engine import make_signal, print_signal

position_mgr = PositionManager()

# 1️⃣ 启动时加载真实仓位
account = position_mgr.load_from_yaml("config/live_positions.yaml")

risk_mgr = RiskManager(
    risk_per_trade=0.01,   # 单笔最多亏 1%
    min_rr=1.5,            # 最低风险回报比
    min_stop_pct=0.01,     # 最小止损 1%
    max_stop_pct=0.03,     # 最大止损 3%
    min_take_pct=0.01,     # 最小止盈 1%
    atr_stop_mult=1.2,     # ATR 止损
    atr_take_mult=2.0,     # ATR 止盈
    lot_size=100,          # A 股
)

# 2️⃣ 每次行情 / 预测更新
def on_bar(ticker, name, price, low, median, high, atr):
    raw_signal = make_signal(
        low=low,
        median=median,
        high=high,
        last_price=price,
    )
   
    final_signal = debouncer_manager.update(ticker, raw_signal)
  
    plan = None
    
    low_v  = float(low[-1])
    high_v = float(high[-1])
    print('price,low, high', price, low_v,high_v)
    plan = risk_mgr.evaluate(
        last_price=price,
        chronos_low=low_v,
        chronos_high=high_v,
        atr=atr,
        capital=account,
    )
  
    order = position_mgr.on_signal(
        ticker=ticker,
        signal=final_signal,
        last_price=price,
        trade_plan=plan,
    )

    if order:
        print("📌 实盘决策:", order)
