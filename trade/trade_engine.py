from venv import logger

import numpy as np
from pandas import DataFrame, Series
from global_state import equity_engine
from infra.core.context import TradingSession
from log import signal_log, risk_log
from predict.predict_result import PredictionResult
from risk.risk_manager import risk_mgr
from strategy.decision_builder import DecisionContextBuilder
from strategy.equity_policy import TradeIntent
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from strategy.signal_mgr import SignalManager
from trade.equity_executor import execute_equity_action
from strategy.equity_policy import TradeIntent
from config.settings import ticker_name_map
from strategy.gate import gater
from plot.decision_debugger import DecisionDebugger

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
"""
        模型预测
    ↓
    score 计算（连续）
    ↓
    Debouncer（事件过滤）
    ↓
    EquityDecision（语义化决策）
    ↓
    PositionManager（下单 / 仓位变化）
    名称	含义	是否连续
    raw_score	模型方向 + 强度	✅ 连续
    confidence	是否触发交易事件	❌ 事件型
"""

#debugger = DecisionDebugger(watch_tickers=['sz300142'])
debugger = DecisionDebugger()
# 2️ 实盘主循环,每次行情 / 预测更新
def execute_stock_decision(
    *,
    ticker: str,
    close_df: DataFrame | Series,
    pre_result: PredictionResult,
    session: TradingSession,
) -> dict:
    """
    每次行情/预测更新，执行单只股票的交易决策
    返回 dict，用于动态表格显示
    """
    position_mgr = session.position_mgr
    # ===== 1️⃣ 最新价格 =====
    price = float(close_df.iloc[-1])
    position_mgr.update_price(ticker, price)

    # ===== 2️⃣ 模型预测 + 信号处理 =====
    ctx_builder = DecisionContextBuilder(
        equity_engine=equity_engine, gater=gater, position_mgr=position_mgr
    )

    signal_mgr = SignalManager(
        debouncer=debouncer_manager,
        min_score=0.08,
    )
    low, median, high, atr = pre_result.low, pre_result.median, pre_result.high, pre_result.atr
    predicted_up = equity_engine.calc_predicted_up_risk_adjusted(
        low, median, high, price
    )

    model_score = float(np.clip(predicted_up, -1.0, 1.0))

    ctx = ctx_builder.build(
        ticker=ticker,
        low=low,
        median=median,
        high=high,
        latest_price=price,
        atr=atr,
        model_score=model_score,
        eq_feat=session.eq_feat,
        close_df=close_df        
    )
    #signal_log(ctx)
    debugger.update(ctx)
    intent = signal_mgr.evaluate(ctx)

    # signal_log(f"{name}/{ticker}: {intent.action} ")
    # signal_log(intent)
    pos_dict = position_mgr.pos_to_dict(ticker=ticker)
    # ===== 3️⃣ 非确认信号 + 非强制减仓直接返回 =====
    if not intent.confirmed and not intent.force_reduce:
        return {
            "ticker": ticker,
            **pos_dict,
            **intent.__dict__,
            "action": "HOLD",
        }

    # ===== 4️⃣ Risk + Budget =====
    low_v = float(pre_result.low[-1])
    high_v = float(pre_result.high[-1])
    position_value = position_mgr.position_value()

    signal_capital = budget_mgr.get_budget(
        ticker=ticker,
        gate_score=intent.gate_mult,
        available_cash=position_mgr.available_cash,
        equity=position_mgr.equity,
        positions_value=position_value,
    )

    risk_log(f"{ticker} budget={signal_capital:.2f} gate={intent.gate_mult:.2f}")

    plan = risk_mgr.evaluate(
        last_price=price,
        chronos_low=low_v,
        chronos_high=high_v,
        atr=pre_result.atr,
        capital=signal_capital,
    )

    if plan is None:
        risk_log(f"{ticker} no risk plan")
        return {
            "ticker": ticker,
            **pos_dict,
            **intent.__dict__,
            "action": "HOLD",
        }

    # ===== 5️⃣ Signal → Trade Action（执行仓位变化）=====
    ret_dict = execute_equity_action(
        decision=intent,
        position_mgr=position_mgr,
        ticker=ticker,
        last_price=price,
    )

    # 返回用于动态表格显示的 dict
    return ret_dict
