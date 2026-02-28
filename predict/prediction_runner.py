# for backtest
import pandas as pd
from infra.core.context import TradingSession
from trade.processor import execute_stock_analysis
from infra.core.runtime import RunMode
from config.settings import TICKER_PERIOD, ticker_name_map
import traceback

def generate_prediction_df(tickers, trade_session, period=TICKER_PERIOD):
    """
    返回一个 DataFrame, 每行是 PredictionResult-like dict
    ticker: list[str]
    period: 数据周期
    """
  
    records = []

    for ticker in tickers:
        try:
            result = execute_stock_analysis(ticker, trade_session)
            print('result', result)
            decision = result["decision"]
            print('decision', decision)
            # 提取必要的预测数据
            record = {
                "ticker": ticker,
                **decision,
                "low": result["low"][-1],
                "median": result["median"][-1],
                "high": result["high"][-1],
            }
            records.append(record)

        except Exception as e:
            print(f"[WARN] {ticker} failed: {e}")
            traceback.print_exc()
    # 转成 DataFrame，每列是 list 对象（和你的 execute_stock_decision 兼容）
    df = pd.DataFrame(records)
    return df