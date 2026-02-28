from equity.equity_features import equity_features
from infra.core.runtime import RunMode
from predict.predict_result import PredictionResult
from trade.trade_engine import execute_stock_decision
import numpy as np
from infra.core.context import TradingSession

class BacktestRunner:
    def __init__(self):
        pass

    def run(self, ticker, market_df, prediction_df, trade_session,mode="original"):
        """
        ticker: 股票代码
        market_df: 包含收盘价、volume等行情信息
        prediction_df: 模型预测结果
        mode: original / ai_mode
        """
        # 初始化 session, position_mgr, equity_engine ...
        
        position_mgr = trade_session.position_mgr

        equity_curve = []

        # 遍历每个时间点
        for i, idx in enumerate(market_df.index):
            # 行情：时间索引
            #close_df = market_df['close']     # DataFrame，1 行
            close_df = market_df['close'].iloc[: i + 1]
            ret = execute_stock_decision(
                ticker=ticker,
                close_df=close_df,
                pre_result=prediction_df,
                session=trade_session,
               # mode=mode,
            )
            # 记录当前净值
            equity_curve.append(position_mgr.equity)

        # --- Stage 2: 自动指标计算 ---
        equity_array = np.array(equity_curve)
        final_equity = equity_array[-1]

        # 最大回撤
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_dd = drawdown.min()

        # 简单收益波动比 (sharpe_like)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_like = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(len(returns))

        return {
            "equity_curve": equity_curve,
            "final_equity": final_equity,
            "max_drawdown": max_dd,
            "sharpe_like": sharpe_like
        }

   