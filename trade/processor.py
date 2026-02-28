# trade/processor.py
from datetime import datetime
import pandas as pd
from data.loader import load_stock_df
from predict.chronos_predict import run_prediction
from predict.time_utils import build_future_index
from strategy.equity_policy import TradeIntent
from trade.trade_engine import execute_stock_decision
from infra.core.context import TradingSession
from predict.prediction_store import update_prediction_history
from config.settings import ticker_name_map


def execute_stock_analysis(
    ticker: str, session: TradingSession
):
    """
    只处理交易逻辑，不涉及 UI 绘图
    """
    period = session.period
    hs300_df = session.hs300_df
    eq_feat = session.eq_feat
    df = load_stock_df(ticker, period)
    # 模型预测
    pre_result = run_prediction(
        df=df, hs300_df=hs300_df, ticker=ticker, period=period, eq_feat=eq_feat
    )
   
    # 执行交易决策
    decision = execute_stock_decision(
        ticker=ticker,
        close_df=df["close"],
        pre_result=pre_result,
        session=session        
    )
    print('decision',decision)
    future_index = build_future_index(df, period)

    history_pred = update_prediction_history(ticker, future_index, pre_result)

    return {
        "ticker": ticker,
        "name": ticker_name_map.get(ticker, ticker),
        "df": df,
        "low": pre_result.low,
        "median": pre_result.median,
        "high": pre_result.high,
        "model_score": pre_result.model_score,
        "future_index": future_index,
        "history_pred": history_pred,
        "last_price": df["close"].iloc[-1],
        "decision": decision,
    }
