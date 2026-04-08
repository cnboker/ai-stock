from pandas import DataFrame
from infra.core.runtime import GlobalState
from infra.core.trade_session import TradingSession
from infra.utils.time_profile import timer_decorator
from log import signal_log, risk_log
from predict.chronos_predict import run_prediction
from risk.risk_manager import risk_mgr
from strategy.decision_builder import DecisionContextBuilder
from strategy.signal_debouncer import debouncer_manager
from risk.budget_manager import budget_mgr
from strategy.signal_mgr import SignalManager
from strategy.gate import gater
from plot.decision_debugger import DecisionDebugger
from trade.trade_system import TradingSystem


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
    每次行情/预测更新，执行单只股票的交易决策
    返回 dict，用于动态表格显示
    """

    # 1. 首先明确“当前”要处理的参考对象（即个股数据）
    hs300_df = hs300_df.iloc[-GlobalState.chronos_context_length:]
    ticker_df = ticker_df.iloc[-GlobalState.chronos_context_length:]
    # ========== HS300 对齐 ==========
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

    # 模型预测
    pre_result = run_prediction(
        df=ticker_df,
        hs300_df=hs300_series,
        ticker=ticker,
        period=session.period,
        eq_feat=session.eq_feat,
    )
    #print(f"pre_result={pre_result}")
    position_mgr = session.position_mgr
    # ===== 1️⃣ 最新价格 =====
    #price = float(ticker_df["close"].iloc[-1])
    price = GlobalState.tickers_price[ticker]
    position_mgr.update_price(ticker, price)

    # ===== 2️⃣ 模型预测 + 信号处理 =====
    ctx_builder = DecisionContextBuilder(gater=gater, position_mgr=position_mgr)
    signal_mgr = SignalManager(
        debouncer=debouncer_manager,
        min_score=0.03,
    )
    tradeSystem = TradingSystem(
        ctx_builder=ctx_builder,
        signal_mgr=signal_mgr,
        budget_mgr=budget_mgr,
        risk_mgr=risk_mgr,
        position_mgr=position_mgr,
    )
    return tradeSystem.run_tick(
        ticker=ticker,
        pre_result=pre_result,
        close_df=ticker_df["close"],
        session=session,
    )
