from venv import logger

from pandas import DataFrame
from position.position_manager import position_mgr
from risk.risk_manager import risk_mgr
from strategy.gate import gater
from strategy.signal_debouncer import debouncer_manager
from strategy.signal_engine import make_signal, print_signal
from risk.BudgetManager import budget_mgr

'''
Chronos 区间
   ↓
PredictionGate   ←【只做一件事：值不值得信】
   ↓
make_signal
   ↓
debouncer
   ↓
risk_mgr
   ↓
position_mgr

Gate 决定“值不值得冒险”
Risk 决定“冒多少险”
PositionManager 决定“钱够不够”

'''

# 2️ 实盘主循环,每次行情 / 预测更新
def on_bar(ticker, name, context: DataFrame, low, median, high, atr):
    price = context.iloc[-1]
    
    gate_result = gater.evaluate(
        lower=low,
        mid=median,
        upper=high,
        context=context.values,
        # y_proxy=y_proxy,  # 回测用真实，实盘可不传
    )

    if not gate_result.allow:
        raw_signal = "HOLD"
    else:
        raw_signal = make_signal(
            low=low,
            median=median,
            high=high,
            last_price=price,
        )

    final_signal = debouncer_manager.update(ticker, raw_signal)
    print_signal(f"{name}[{ticker}]", final_signal)
    plan = None

    low_v = float(low[-1])
    high_v = float(high[-1])
    # print('price,low, high,atr', price, low_v,high_v,atr)

    #投资最大仓位,不是“直接下单的钱”，而是「这一次信号允许你冒险的资金预算」
    # 计算预算

    position_value = position_mgr.market_value(ticker=ticker,latest_price=price)
    print('position_value', position_value)
    signal_capital = budget_mgr.get_budget(
        ticker=ticker,
        gate_score=gate_result.score,
        available_cash=position_mgr.available_cash,
        equity=position_mgr.equity,
        positions_value=position_value,
    )
    print("允许你冒险的资金预算", signal_capital)
    print("chronos_low,chronos_high", low_v,high_v)
    plan = risk_mgr.evaluate(
        last_price=price,
        chronos_low=low_v,
        chronos_high=high_v,
        atr=atr,
        capital=signal_capital,
    )
    print("风险计划", plan)
    order = position_mgr.on_signal(
        ticker=ticker,
        signal=final_signal,
        last_price=price,
        trade_plan=plan,
    )
  
    if order:
        print("📌 实盘决策:", order)
