from venv import logger

from pandas import DataFrame
from log import order_log_1, risk_log, signal_log
from position.position_manager import position_mgr
from risk.risk_manager import risk_mgr
from strategy.gate import gater
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from position.PositionPolicy import position_policy
from strategy.signal_mgr import SignalManager

"""
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

"""


# 2️ 实盘主循环,每次行情 / 预测更新
def on_bar(
    ticker: str,
    name: str,
    context: DataFrame,
    low,
    median,
    high,    
    atr: float,
    model_score,
    eq_feat
):
    # ===== 1. 最新价格 =====
    price = float(context.iloc[-1])
    position_mgr.update_price(ticker, price)
   
    signal_mgr = SignalManager(
        gater=gater,
        debouncer_manager=debouncer_manager,
        min_score=0.08,
    )    
    has_position = ticker in position_mgr.positions
    final_action, confidence, gate_result = signal_mgr.evaluate(
        ticker=ticker,
        low=low,
        median=median,
        high=high,
        latest_price=price,
        context=context,
        model_score=model_score,
        eq_feat=eq_feat,
        has_position=has_position,
        atr=atr
    )

    signal_log(f"{name}/{ticker}: {final_action} (confidence={confidence:.2f})")

    # =====Risk + Budget =====
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

    risk_log(f"{ticker} budget={signal_capital:.2f} " f"gate={gate_result.score:.2f}")

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

    # ===== 5. Signal → Trade Action（开仓信号）=====
    action = position_mgr.on_signal(
        symbol=ticker,
        action=final_action,
        confidence=confidence,
        last_price=price,
        trade_plan=plan,
    )

      # 4️⃣ Gate Reject → Policy 干预(平仓或建仓)
    if action is None:
        position = position_mgr.positions.get(ticker)
        if position:
            action = position_policy.decide(position, gate=gate_result)  # gate_result 可选

    # 5️⃣ 执行动作
    if action:
        position_mgr.apply_action(ticker, action)
        position = position_mgr.positions.get(ticker)
        if position:
            order_log_1(ticker, action=action, position=position)
   