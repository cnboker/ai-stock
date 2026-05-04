import time

from pandas import DataFrame
from infra.core.runtime import GlobalState
from infra.core.trade_session import TradingSession
from infra.utils.time_profile import timer_decorator
from log import signal_log, risk_log
from predict.inference import run_prediction
from risk.risk_manager import risk_mgr
from strategy.decision_builder import DecisionContextBuilder
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from strategy.signal_mgr import SignalManager
from strategy.gate import gater
from plot.decision_debugger import DecisionDebugger
from trade.trade_system import TradingSystem
from infra.core.dynamic_settings import settings

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
# @timer_decorator, 如果想测单次决策耗时，可以打开这个装饰器：0.04s 左右，主要耗在模型推理上
def execute_stock_decision(
    *,
    ticker: str,
    hs300_df: DataFrame,
    ticker_df: DataFrame,
    session: TradingSession,
) -> dict:
    """
    运行预测与构建 Context，但不执行交易，返回候选信息
    """
    # 1. HS300 与个股数据对齐
    hs300_df = hs300_df.iloc[-settings.CHRONOS_CONTEXT_LENGTH :]
    ticker_df = ticker_df.iloc[-settings.CHRONOS_CONTEXT_LENGTH :]

    if hs300_df is not None and not hs300_df.empty:
        hs300_series = (
            hs300_df["close"]
            .reindex(ticker_df.index, method="nearest")
            .interpolate("linear")
            .ffill()
            .values
        )
    else:
        hs300_series = ticker_df["close"].values

    # 2. Kronos 模型预测 (获取包含 std 的结果)
    
    pre_result = run_prediction(
        df=ticker_df,
        hs300_df=hs300_series,
        ticker=ticker,
        period=session.period,
        eq_feat=session.eq_feat,
    )
   
    # 3. 更新价格状态
    price = GlobalState.tickers_price[ticker]
    session.position_mgr.update_price(ticker, price)

    # 4. 构建上下文 (引入 std 风险因子)
    ctx_builder = DecisionContextBuilder(gater=gater, position_mgr=session.position_mgr)
    ctx = ctx_builder.build(
        ticker=ticker,
        low=pre_result.low,
        median=pre_result.median,
        high=pre_result.high,
        std=pre_result.std,
        atr=pre_result.atr,
        model_score=pre_result.model_score,
        close_df=ticker_df["close"],
        eq_decision=session.tradeIntent,
    )

    # 2. 【核心改动】判断是否属于“必须立即执行”的退出信号
    # 只要是卖出类信号（减仓、清仓、止损），不进漏斗，直接执行
    if ctx.raw_signal in ("REDUCE", "LIQUIDATE") and session.position_mgr.has_position(ticker):
        signal_log(f"🚨 {ticker} 触发退出信号: {ctx.raw_signal} | 原因: {ctx.liquidate_reason}")
        
        # 立即调用执行函数
        execute_final_order({
            "ticker": ticker,
            "ctx": ctx,
            "ticker_df": ticker_df,
            "low": pre_result.low,
            "high": pre_result.high,
        }, session.position_mgr)
        
        return {"type": "exit", "ticker": ticker}

    # 3. 如果是买入信号，则返回给漏斗
    # signal_log(f"🔍 {ticker} predicted_up:{ctx.predicted_up:.3f} | 模型预测分数: {ctx.model_score:.4f} | 信号: {ctx.raw_signal} | 价格: {price:.2f},gate_allow:{ctx.gate_allow}")
    if ctx.raw_signal == "LONG" and ctx.gate_allow:
        signal_log(f"✅ {ticker} 通过漏斗，准备执行买入 | 模型分数: {ctx.model_score:.4f} | 价格: {price:.2f}")
        return {
            "type": "candidate",
            "ticker": ticker,
            "ctx": ctx,
            "std": pre_result.std,
            "ticker_df": ticker_df,
            "low": pre_result.low,
            "median": pre_result.median,
            "high": pre_result.high,
        }

    return {"type": "none", "ticker": ticker}


def execute_final_order(candidate: dict, position_mgr) -> None:
    """
    接收优选后的候选字典，执行 TradingSystem 逻辑
    """
    rank_score = candidate.get("rank_score", 0)
    if rank_score <= 0:
        return  # 预测涨幅为负，直接不进入漏斗
    ticker = candidate["ticker"]
    ctx = candidate["ctx"]
    # ticker_df = candidate["ticker_df"]
    low = candidate["low"]
    # median = candidate["median"]
    high = candidate["high"]
    
    # 保持原有组件初始化
    signal_mgr = SignalManager(
        debouncer=debouncer_manager,
        min_score=0.03,
    )

    # 构建交易系统并执行 tick 动作
    trade_system = TradingSystem(
        signal_mgr=signal_mgr,
        budget_mgr=budget_mgr,
        risk_mgr=risk_mgr,
        position_mgr=position_mgr,
    )

    return trade_system.run_tick(
        ticker=ticker,
        ctx=ctx,  # 注意：此时 run_tick 应该直接接收构建好的 ctx
        low=low,
        high=high,
    )
