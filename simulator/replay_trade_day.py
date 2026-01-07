import pandas as pd
from data.loader import load_stock_df
from config.settings import TICKER_PERIOD
from predict.chronos_predict import run_prediction
from predict.predict_result import PredictionResult
from strategy.equity_policy import TradeIntent
from trade.trade_engine import execute_stock_decision
from infra.core.context import TradingContext
from infra.core.runtime import RunMode
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from data.loader import load_index_df
from simulator.anysis import dump_abnormal_predictions
from dataclasses import dataclass

@dataclass
class EvalResult:
    prediction: PredictionResult
    decision: TradeIntent


def simulate_trade_day(ticker: str, trade_date: str, period="3"):

    trade_day = pd.to_datetime(trade_date)

    df_all = load_stock_df(
        ticker=ticker,
        period=TICKER_PERIOD,        
    )

    if df_all.empty:
        raise ValueError(f"{ticker} 无历史数据")

    hs300_df = load_index_df(period)
    
    context = TradingContext(
        run_mode=RunMode.SIM,
        position_mgr=create_position_manager(10000, RunMode.SIM),
        eq_recorder=create_equity_recorder(RunMode.SIM, ticker),
        ticker=ticker,
        period=period,
        hs300_df=hs300_df
    )
    
    eq_feat = context.eq_feat
    # 当天数据
    df_today = df_all[df_all.index.date == trade_day.date()]
    df_history = df_all[df_all.index.date < trade_day.date()]

    if df_today.empty:
        raise ValueError(f"{ticker} {trade_date} 当天无数据")

    print(f"[SIM] 开始回放 {ticker} {trade_date} 共 {len(df_today)} 根 3min")
   
    results = []
    for i in range(len(df_today)):
        df_slice = df_today.iloc[: i + 1]
        df_combined = pd.concat([df_history, df_slice])
        #print('df_combined',df_combined)
        # ===== 预测 =====
        pre_result = run_prediction(
            df=df_combined,
            hs300_df=hs300_df,     # 模拟盘可以先关
            ticker=ticker,
            period=period,
            eq_feat=eq_feat       # 或者传你已有的 eq_feat
        )
        
        # with open("pre_result.txt", "a", encoding="utf-8") as f:
        #     f.write(str(pre_result) + "\n\n")  # 每条记录之间空一行
        
        # ===== 实盘核心逻辑（完全复用）=====
        decision = execute_stock_decision(
            close_df=df_slice["close"],
            pre_result=pre_result,
            context=context
        )
        result = EvalResult(
            prediction=pre_result,
            decision=decision
        )
        results.append(result)
        context.eq_recorder._save_disk()

    
    dump_abnormal_predictions(results)