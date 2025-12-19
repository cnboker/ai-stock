from venv import logger
from position.position_manager import position_mgr
from risk.risk_manager import risk_mgr
from strategy.signal_debouncer import debouncer_manager
from strategy.signal_engine import make_signal, print_signal


# 2️ 实盘主循环,每次行情 / 预测更新
def on_bar(ticker, name, price, low, median, high, atr):
    raw_signal = make_signal(
        low=low,
        median=median,
        high=high,
        last_price=price,
    )
   
    final_signal = debouncer_manager.update(ticker, raw_signal)
    print_signal(f"{name}[{ticker}]",final_signal)
    plan = None
    
    low_v  = float(low[-1])
    high_v = float(high[-1])
    #print('price,low, high,atr', price, low_v,high_v,atr)
    plan = risk_mgr.evaluate(
        last_price=price,
        chronos_low=low_v,
        chronos_high=high_v,
        atr=atr,
        capital=position_mgr.account,
    )
  
    order = position_mgr.on_signal(
        ticker=ticker,
        signal=final_signal,
        last_price=price,
        trade_plan=plan,
    )

    if order:
        print("📌 实盘决策:", order)
