import numpy as np
from pandas import DataFrame, Series
from global_state import equity_engine
from infra.core.context import TradingSession
from infra.persistence.live_positions import persist_live_positions
from log import signal_log, risk_log
from predict.predict_result import PredictionResult
from risk.risk_manager import risk_mgr
from strategy.decision_builder import DecisionContextBuilder
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from strategy.signal_mgr import SignalManager
from trade.equity_executor import execute_equity_action
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

# debugger = DecisionDebugger(watch_tickers=['sz300142'])
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
        min_score=0.03,
    )
    low, median, high, atr, model_score = (
        pre_result.low,
        pre_result.median,
        pre_result.high,
        pre_result.atr,
        pre_result.model_score,
    )

    ctx = ctx_builder.build(
        ticker=ticker,
        low=low,
        median=median,
        high=high,
        latest_price=price,
        atr=atr,
        model_score=model_score,
        eq_feat=session.eq_feat,
        close_df=close_df,
        eq_decision=session.tradeIntent,
        
    )
    session.eq_recorder.add(position_mgr.equity)

    if ctx.raw_signal == "SHORT":
        return None
    # signal_log(ctx)

    # debugger.update(ctx)

        # 1. 评估信号意图
    intent = signal_mgr.evaluate(ctx)

    # 2. 关键：在被 Confirmed 逻辑拦截前，记录信号状态
    # 这样你就能看到为什么 count/confirm_n 没达标
    # signal_log(f"[{ticker}] Score: {ctx.raw_score:.3f} | Bucket: {intent.action} | Count: {intent.count}/{intent.confirm_n} | Confirmed: {intent.confirmed}")

    # 3. 拦截未确认信号
    if not intent.confirmed and not intent.force_reduce:
        # 如果没确认，但你有仓位，可能需要维持当前持仓状态（防止被当作信号丢失处理）
        persist_live_positions(position_mgr)
        return {
            "ticker": ticker,
            "action": "HOLD",
            #"reason": f"Waiting for confirmation ({intent.count}/{intent.confirm_n})"
        }

    # 4. 只有 Confirmed == True 才会运行到这里
    plan = None
    if not position_mgr.has_position(ticker):
        # 计算预算
        signal_capital = budget_mgr.get_budget(
            ticker=ticker,
            gate_score=intent.gate_mult,
            available_cash=position_mgr.available_cash,
            equity=position_mgr.equity,
            positions_value=position_mgr.position_value(),
        )

    
        # 风险评估与下单计划
        plan = risk_mgr.evaluate(
            ticker=ticker,
            last_price=price,
            chronos_low=float(pre_result.low[-1]),
            chronos_high=float(pre_result.high[-1]),
            atr=pre_result.atr,
            capital=signal_capital,
            position_mgr=position_mgr,
        )
        # 这行日志现在会在每一根满足条件的 K 线上打印
        signal_log(f"🔥 [EXECUTE] {ticker} budget={signal_capital:.2f} gate={intent.gate_mult:.2f} plan={plan}")

    # ===== 5️⃣ Signal → Trade Action（执行仓位变化）=====
    ret_dict = execute_equity_action(
        decision=intent,
        position_mgr=position_mgr,
        ticker=ticker,
        last_price=price,
        plan=plan,
    )
    persist_live_positions(position_mgr)
    # 返回用于动态表格显示的 dict
    return ret_dict
