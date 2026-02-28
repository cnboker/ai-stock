# main.py
from huggingface_hub import get_session
import pandas as pd
import matplotlib.pyplot as plt
from backtest.runner import BacktestRunner
from config.settings import TICKER_PERIOD
from global_state import equity_engine
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from infra.core.context import TradingSession
from infra.core.runtime import RunMode
from position.position_factory import create_position_manager
from predict.prediction_runner import generate_prediction_df
from data.loader import load_index_df, load_stock_df


def init_session(period=TICKER_PERIOD):
        """
        初始化 TradingSession，用于回测
        """
        hs300_df = load_index_df(str(period))
        position_mgr = create_position_manager(0, RunMode.BACKTEST)
        eq_recorder = create_equity_recorder(RunMode.BACKTEST)
        eq_feat = equity_features(eq_recorder.to_series())
        tradeIntent = equity_engine.decide(eq_feat,position_mgr.has_any_position())
        session = TradingSession(
            run_mode=RunMode.BACKTEST,
            period=str(period),
            hs300_df=hs300_df,
            eq_feat=eq_feat,
            tradeIntent=tradeIntent,
            position_mgr=position_mgr,
            eq_recorder=eq_recorder,
        )

        return session

# 假设我们回测的股票池
tickers = ["sz300142", "sz300143"]

# 加载历史行情数据
market_data = {ticker: load_stock_df(ticker, period=TICKER_PERIOD) for ticker in tickers}  # 示例
trade_session = init_session()
# 生成 prediction_df
prediction_df = generate_prediction_df(tickers,trade_session, period=TICKER_PERIOD)

print('prediction_df->',prediction_df)

runner = BacktestRunner()

# 原策略
result_old = runner.run("sz300142", market_data["sz300142"], prediction_df, trade_session, mode="original")

# # AI 模式策略
# result_ai = runner.run("sz300142", market_data["sz300142"], prediction_df, trade_session, mode="ai_mode")

# print("Old Strategy:", result_old["final_equity"], result_old["max_drawdown"], result_old["sharpe_like"])
# print("AI Mode Strategy:", result_ai["final_equity"], result_ai["max_drawdown"], result_ai["sharpe_like"])

# # 绘制对比曲线
# plt.plot(result_old["equity_curve"], label="Old Strategy")
# plt.plot(result_ai["equity_curve"], label="AI Mode Strategy")
# plt.legend()
# plt.title("Equity Curve Comparison")
# plt.grid(True)
# plt.show()

