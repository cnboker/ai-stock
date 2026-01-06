from venv import logger

from pandas import DataFrame
from equity import equity_engine
from infra.core.context import TradingContext
from log import signal_log,risk_log
from predict.predict_result import PredictionResult
from risk.risk_manager import risk_mgr
from strategy.equity_policy import TradeContext
from strategy.gate import gater
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from strategy.signal_mgr import SignalManager
from trade.equity_executor import execute_equity_action

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
'''
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
'''

# 2️ 实盘主循环,每次行情 / 预测更新
def execute_stock_decision(
    *,    
    close_df: DataFrame,
    pre_result: PredictionResult,    
    context: TradingContext
) -> dict:
    """
    每次行情/预测更新，执行单只股票的交易决策
    返回 dict，用于动态表格显示
    """
    ticker = context.ticker
    name = context.name
    position_mgr = context.position_mgr

    # ===== 1️⃣ 最新价格 =====
    price = float(close_df.iloc[-1])
    position_mgr.update_price(ticker, price)
   
    # ===== 2️⃣ 模型预测 + 信号处理 =====
    signal_mgr = SignalManager(
        gater=gater,
        debouncer_manager=debouncer_manager,        
        min_score=0.08,
    )    
    has_position = position_mgr.get(ticker) is not None

    decision: TradeContext = signal_mgr.evaluate(
        ticker=ticker,
        low=pre_result.low,
        median=pre_result.median,
        high=pre_result.high,
        latest_price=price,
        close_df=close_df,
        model_score=pre_result.model_score,
        eq_feat=context.eq_feat,
        has_position=has_position,
        atr=pre_result.atr
    )

    signal_log(
        f"{name}/{ticker}: {decision.action} "
        f"(conf={decision.confidence:.2f}, reason={decision.reason})"
    )

    # ===== 3️⃣ 非确认信号 + 非强制减仓直接返回 =====
    if not decision.confirmed and not decision.force_reduce:
        return {
            "ticker": ticker,
            "position": position_mgr.get(ticker),
            "action": "HOLD",
            "confidence": decision.confidence,
            "model_score": decision.model_score,
            "atr": decision.atr,
            "predicted_up": getattr(decision, "predicted_up", False),
            "raw_score": getattr(decision, "raw_score", 0),
        }

    # ===== 4️⃣ Risk + Budget =====
    low_v = float(pre_result.low[-1])
    high_v = float(pre_result.high[-1])
    position_value = position_mgr.position_value()

    signal_capital = budget_mgr.get_budget(
        ticker=ticker,
        gate_score=decision.gate_mult,
        available_cash=position_mgr.available_cash,
        equity=position_mgr.equity,
        positions_value=position_value,
    )

    risk_log(f"{ticker} budget={signal_capital:.2f} gate={decision.gate_mult:.2f}")

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
            "position": position_mgr.get(ticker),
            "action": "HOLD",
            "confidence": decision.confidence,
            "model_score": decision.model_score,
            "atr": decision.atr,
            "predicted_up": getattr(decision, "predicted_up", False),
            "raw_score": getattr(decision, "raw_score", 0),
        }

    # ===== 5️⃣ Signal → Trade Action（执行仓位变化）=====
    ret_dict = execute_equity_action(
        decision=decision,
        position_mgr=position_mgr,
        ticker=ticker,
        last_price=price,
    )

    # 返回用于动态表格显示的 dict
    return ret_dict
