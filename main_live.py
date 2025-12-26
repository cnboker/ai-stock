from venv import logger

from pandas import DataFrame
from log import order_log, risk_log, signal_log
from position.position_manager import position_mgr
from risk.risk_manager import risk_mgr
from strategy.gate import gater
from strategy.signal_debouncer import debouncer_manager
from strategy.signal_engine import make_signal, print_signal
from risk.BudgetManager import budget_mgr
from position.PositionPolicy import position_policy

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
def on_bar(
    ticker: str,
    name: str,
    context: DataFrame,
    low,
    median,
    high,
    atr: float,
):
    print('test')
    # ===== 1. 最新价格 =====
    price = float(context.iloc[-1])
    position_mgr.update_price(ticker, price)
    
    # ===== 2. Gate =====
    gate_result = gater.evaluate(
        lower=low,
        mid=median,
        upper=high,
        context=context.values,
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

    # ===== 3. Debounce =====
    final_signal = debouncer_manager.update(ticker, raw_signal)
    signal_log(f"{name}/{ticker}: {final_signal}")

    # ===== 4. Risk + Budget =====
    low_v = float(low[-1])
    high_v = float(high[-1])

    position_value = position_mgr.position_value()

    signal_capital = budget_mgr.get_budget(
        ticker=ticker,
        gate_score=gate_result.score,
        available_cash=position_mgr.available_cash,
        equity=position_mgr.equity,
        positions_value=position_value,
    )

    risk_log(
        f"{ticker} budget={signal_capital:.2f} "
        f"gate={gate_result.score:.2f}"
    )

    plan = risk_mgr.evaluate(
    last_price=price,
    chronos_low=low_v,
    chronos_high=high_v,
    atr=atr,
    capital=signal_capital,
)

    if plan is None:
        risk_log(f"{ticker} no risk plan")
        return

    # ===== 5. Signal → Trade Action（唯一 OPEN 来源）=====
    action = position_mgr.on_signal(
        symbol=ticker,
        signal=final_signal,
        last_price=price,
        trade_plan=plan,
    )

    # ===== 6. Gate Reject → Policy 干预 =====
    if action is None and not gate_result.allow:
        position = position_mgr.positions.get(ticker)
        action = position_policy.decide(position, gate_result)

    # ===== 7. 执行 =====
    if action:
        position_mgr.apply_action(ticker, action)
